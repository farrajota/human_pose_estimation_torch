--[[
    Helpful functions for evaluation.
]]


function calcDistsOne(preds, label, normalize)
    local dists = torch.Tensor(preds:size(1))
    for i = 1,preds:size(1) do
        if label[i][1] > 1 and label[i][2] > 1 then
            dists[i] = torch.dist(label[i],preds[i])/normalize
        else
            dists[i] = -1
        end
    end
    return dists
end

------------------------------------------------------------------------------------------------------------

function calcDists(preds, label, normalize)
    local dists = torch.Tensor(preds:size(2), preds:size(1))
    local diff = torch.Tensor(2)
    for i = 1,preds:size(1) do
        for j = 1,preds:size(2) do
            if label[i][j][1] > 1 and label[i][j][2] > 1 then
                dists[j][i] = torch.dist(label[i][j],preds[i][j])/normalize[i]
            else
                dists[j][i] = -1
            end
        end
    end
    return dists
end

------------------------------------------------------------------------------------------------------------

local function fitParabola(x1,x2,x3,y1,y2,y3)
  local x1_sqr = x1*x1
  local x2_sqr = x2*x2
  local x3_sqr = x3*x3

  local div = (x1_sqr-x1*(x2+x3)+x2*x3)*(x2-x3)
  local a = (x1*(y2-y3)-x2*(y1-y3)+x3*(y1-y2))/div
  local b = (x1_sqr*(y2-y3)-x2_sqr*(y1-y3)+x3_sqr*(y1-y2))/div

  return b/(2*a)
end

local function fitParabolaAll(hms, coords)
    local preds = coords:clone():fill(0)
    local nparts = hms:size(2)
    local h,w = hms:size(3),hms:size(4)
    for i=1, hms:size(1) do
        for j=1, nparts do
            local x = {coords[i][j][1]-1, coords[i][j][1], coords[i][j][1]+1}
            local y = {coords[i][j][2]-1, coords[i][j][2], coords[i][j][2]+1}

            if x[1]>=1 and x[3]<=w and y[2]>=1 and y[2]<=h then
                preds[i][j][1] = fitParabola(x[1],x[2],x[3],hms[i][j][y[2]][x[1]],hms[i][j][y[2]][x[2]],hms[i][j][y[2]][x[3]])
            else
                preds[i][j][1]=x[2] -- skip parabola fitting for this coordinate
            end

            if y[1]>=1 and y[3]<=h and x[2]>=1 and x[2]<=w then
                preds[i][j][2] = fitParabola(y[1],y[2],y[3],hms[i][j][y[1]][x[2]],hms[i][j][y[2]][x[2]],hms[i][j][y[3]][x[2]])
            else
                preds[i][j][2]=y[2] -- skip parabola fitting for this coordinate
            end
        end
    end
    return preds
end

------------------------------------------------------------------------------------------------------------

function getPreds(hm)
    assert(hm:dim() == 4, 'Input must be 4-D tensor')
    --local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    --local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    --preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    --preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)

    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local coords_peak = torch.repeatTensor(idx, 1, 1, 2):float()
    coords_peak[{{}, {}, 1}]:apply(function(x) if x%hm:size(4)==0 then return hm:size(4) else return x%hm:size(4) end end)
    coords_peak[{{}, {}, 2}]:div(hm:size(3)):ceil()
    local preds = fitParabolaAll(hm, coords_peak)

    return preds
end

------------------------------------------------------------------------------------------------------------

function getPredsBenchmark(hms, center, scale)
    if hms:dim() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end

   -- -- Get locations of maximum activations (OLD CODE)
   -- local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
   -- local preds = torch.repeatTensor(idx, 1, 1, 2):float()
   -- preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
   -- preds[{{}, {}, 2}]:add(-1):div(hms:size(3)):floor():add(.5)

    -- Get locations of maximum activations (using sub-pixel precision)
    local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
    local coords_peak = torch.repeatTensor(idx, 1, 1, 2):float()
    coords_peak[{{}, {}, 1}]:apply(function(x) return x % hms:size(4) end)
    coords_peak[{{}, {}, 2}]:div(hms:size(3)):ceil()
    local preds = fitParabolaAll(hms, coords_peak)

    -- Get transformed coordinates
    local preds_tf = torch.zeros(preds:size())
    for i = 1,hms:size(1) do        -- Number of samples
        for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
            preds_tf[i][j] = mytransform(preds[i][j],center,scale,0,hms:size(3),true)
        end
    end

    return preds, preds_tf
end

------------------------------------------------------------------------------------------------------------

function distAccuracy(dists, thr)
    -- Return percentage below threshold while ignoring values with a -1
    if not thr then thr = .5 end
    if torch.ne(dists,-1):sum() > 0 then
        return dists:le(thr):eq(dists:ne(-1)):sum() / dists:ne(-1):sum()
    else
        return -1
    end
end

------------------------------------------------------------------------------------------------------------

function heatmapAccuracy(output, label, thr, idxs)
    -- Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
    -- First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    local preds = getPreds(output)
    local gt = getPreds(label)
    local dists = calcDists(preds, gt, torch.ones(preds:size(1))*opt.outputRes/10)
    local acc = {}
    local avgAcc = 0.0
    local badIdxCount = 0

    if not idxs then
        for i = 1,dists:size(1) do
            acc[i+1] = distAccuracy(dists[i], thr)
            if acc[i+1] >= 0 then
                avgAcc = avgAcc + acc[i+1]
            else
                badIdxCount = badIdxCount + 1
            end
        end
        acc[1] = avgAcc / (dists:size(1) - badIdxCount)
    else
        for i = 1,#idxs do
            acc[i+1] = distAccuracy(dists[idxs[i]], thr)
            if acc[i+1] >= 0 then
                avgAcc = avgAcc + acc[i+1]
            else
                badIdxCount = badIdxCount + 1
            end
        end
        acc[1] = avgAcc / (#idxs - badIdxCount)
    end
    return unpack(acc)
end

------------------------------------------------------------------------------------------------------------

function basicAccuracy(output, label, thr)
    -- Calculate basic accuracy
    if not thr then thr = .5 end -- Default threshold of .5
    output = output:view(output:numel())
    label = label:view(label:numel())

    local rounded_output = torch.ceil(output - thr):typeAs(label)
    local eql = torch.eq(label,rounded_output):typeAs(label)

    return eql:sum()/output:numel()
end

------------------------------------------------------------------------------------------------------------

function displayPCK(dists, part_idx, label, title, threshold, show_key)
    -- Generate standard PCK plot
    if not (type(part_idx) == 'table') then
        part_idx = {part_idx}
    end

    local thresh = threshold or 0.5
    local curve_res = 11
    local num_curves = #dists
    local t = torch.linspace(0,1,curve_res)*thresh
    local pdj_scores = torch.zeros(num_curves, curve_res)
    local plot_args = {}
    local results = {}
    print(title)
    for curve = 1,num_curves do
        for i = 1,curve_res do
            --t[i] = (i-1)*.05
            local acc = 0.0
            for j = 1,#part_idx do
                acc = acc + distAccuracy(dists[curve][part_idx[j]], t[i])
            end
            pdj_scores[curve][i] = acc / #part_idx
        end
        plot_args[curve] = {label[curve],t,pdj_scores[curve],'-'}
        print(label[curve],pdj_scores[curve][curve_res])
        results[curve] = {title, label[curve], pdj_scores[curve][curve_res]}
    end

    require 'gnuplot'
    gnuplot.raw('set title "' .. title .. '"')
    if not show_key then gnuplot.raw('unset key')
    else gnuplot.raw('set key font ",6" right bottom') end
    gnuplot.raw(('set xrange [0:%.2f]'):format(thresh))
    gnuplot.raw('set yrange [0:1]')
    gnuplot.plot(unpack(plot_args))

    return results
end

------------------------------------------------------------------------------------------------------------

function accuracy(output,label)
    --local jntIdxs = {mpii={1,2,3,4,5,6,11,12,15,16},flic={2,3,5,6,7,8}}
    local jntIdxs = {
        ['mpii']={1,2,3,4,5,6,10,11,12,13,14,15,16},
        ['flic']={1,2,3,4,5,6},
        ['lsp']={1,2,3,4,5,6,7,8,9,10,11,12,13,14},
        ['mscoco']={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17},
        ['mpii+lsp']={1,2,3,4,5,6,7,8,9,10,11,12,13,14}
    }
    if opt.task == 'pose-int' then
        if type(output) == 'table' then
            return heatmapAccuracy(output[#output],label[#output],nil,jntIdxs[opt.dataset])
        else
            return heatmapAccuracy(output,label,nil,jntIdxs[opt.dataset])
        end
    else
        return heatmapAccuracy(output,label,nil,jntIdxs[opt.dataset])
    end
end