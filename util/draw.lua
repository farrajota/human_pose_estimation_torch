--[[
    Functions used in the demo display.
]]


local function drawLineColor(img,pt1,pt2,width,color)
    -- I'm sure there's a line drawing function somewhere in Torch,
    -- but since I couldn't find it here's my basic implementation
    local color = color or {1,1,1}
    local m = torch.dist(pt1,pt2)
    local dy = (pt2[2] - pt1[2])/m
    local dx = (pt2[1] - pt1[1])/m
    for j = 1,width do
        local start_pt1 = torch.Tensor({pt1[1] + (-width/2 + j-1)*dy, pt1[2] - (-width/2 + j-1)*dx})
        start_pt1:ceil()
        for i = 1,torch.ceil(m) do
            local y_idx = torch.ceil(start_pt1[2]+dy*i)
            local x_idx = torch.ceil(start_pt1[1]+dx*i)
            if y_idx - 1 > 0 and x_idx -1 > 0 and y_idx < img:size(2) and x_idx < img:size(3) then
                img:sub(1,1,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[1])
                img:sub(2,2,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[2])
                img:sub(3,3,y_idx-1,y_idx,x_idx-1,x_idx):fill(color[3])
            end
        end
    end
    img[img:gt(1)] = 1

    return img
end

------------------------------------------------------------------------------------------------------------

function drawSkeletonFLIC(input, hms, coords)

    local im = input:clone()

    local pairRef = {
        --{1,4}, {2,5}, {3,6}, {7,8}
        {1,2}, {2,3},
        {4,5}, {5,6},
        {7,8},
        {9,10}, {10,11}, {9,11}

    }

    local partNames = {'LSho','LElb','LWri','RSho','RElb','RWri',
                       'LHip','RHip', 'LEye', 'REye', 'Nose'}
    local partColor = {1,1,1,2,2,2,3,3,4,4,4}

    local actThresh = 0.00001

    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if hms[pairRef[i][1]]:mean() > actThresh and hms[pairRef[i][2]]:mean() > actThresh then
            -- Set appropriate line color
            local color
            if partColor[pairRef[i][1]] == 1 then color = {.1, 1, .1}
            elseif partColor[pairRef[i][1]] == 2 then color = {1, 0, .3}
            elseif partColor[pairRef[i][1]] == 3 then color = {0, 0, 1}
            elseif partColor[pairRef[i][1]] == 4 then color = {0, .2, 1}
            else color = {.7,0,.7} end

            -- Draw line
            im = drawLineColor(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
        end
    end

    return im
end

------------------------------------------------------------------------------------------------------------

function drawSkeletonLSP(input, hms, coords)

    local im = input:clone()

    local pairRef = {
        {1,2},      {2,3},
        {4,5},      {5,6},
        {7,8},      {8,9},
        {10,11},    {11,12},
        {13,14},
    }

    local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
                       'RWri','RElb','RSho','LSho','LElb','LWri',
                       'Neck','Head'}
    local partColor = {1,1,1,2,2,2,3,3,3,4,4,4,0,0}

    local actThresh = 0.002

    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if hms[pairRef[i][1]]:mean() > actThresh and hms[pairRef[i][2]]:mean() > actThresh then
            -- Set appropriate line color
            local color
            if partColor[pairRef[i][1]] == 1 then color = {0, .3, 1}
            elseif partColor[pairRef[i][1]] == 2 then color = {1, .3, 0}
            elseif partColor[pairRef[i][1]] == 3 then color = {0, 0, 1}
            elseif partColor[pairRef[i][1]] == 4 then color = {1, 0, 0}
            else color = {.7,0,.7} end

            -- Draw line
            im = drawLineColor(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
        end
    end

    return im
end

------------------------------------------------------------------------------------------------------------

function drawSkeletonMPII(input, hms, coords)

    local im = input:clone()

    local pairRef = {
        {1,2},      {2,3},      {3,7},
        {4,5},      {4,7},      {5,6},
        {7,9},      {9,10},
        {14,9},     {11,12},    {12,13},
        {13,9},     {14,15},    {15,16}
    }

    local partNames = {'RAnk','RKne','RHip','LHip','LKne','LAnk',
                       'Pelv','Thrx','Neck','Head',
                       'RWri','RElb','RSho','LSho','LElb','LWri'}
    local partColor = {1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4}

    local actThresh = 0.002

    -- Loop through adjacent joint pairings
    for i = 1,#pairRef do
        if hms[pairRef[i][1]]:mean() > actThresh and hms[pairRef[i][2]]:mean() > actThresh then
            -- Set appropriate line color
            local color
            if partColor[pairRef[i][1]] == 1 then color = {0, .3, 1}
            elseif partColor[pairRef[i][1]] == 2 then color = {1, .3, 0}
            elseif partColor[pairRef[i][1]] == 3 then color = {0, 0, 1}
            elseif partColor[pairRef[i][1]] == 4 then color = {1, 0, 0}
            else color = {.7,0,.7} end

            -- Draw line
            im = drawLineColor(im, coords[pairRef[i][1]], coords[pairRef[i][2]], 4, color, 0)
        end
    end

    return im
end

------------------------------------------------------------------------------------------------------------

function drawSkeleton(input, hms, coords)
    assert(input)
    assert(hms)
    assert(coords)

    local str = string.lower(opt.dataset)
    if str == 'flic' then
        return drawSkeletonFLIC(input, hms, coords)
    elseif str == 'lsp' or str == 'lspe' or str == 'mpii+lsp' then
        return drawSkeletonLSP(input, hms, coords)
    elseif str == 'mpii' then
        return drawSkeletonMPII(input, hms, coords)
    elseif str == 'coco' then
        error('skeleton draw for the COCO dataset is not yet available.')
    else
        error('Invalid dataset: ' .. opt.dataset)
    end
end

------------------------------------------------------------------------------------------------------------

function compileImages(imgs, nrows, ncols, res)
    -- Assumes the input images are all square/the same resolution
    local totalImg = torch.zeros(3,nrows*res,ncols*res)
    for i = 1,#imgs do
        local r = torch.floor((i-1)/ncols) + 1
        local c = ((i - 1) % ncols) + 1
        totalImg:sub(1,3,(r-1)*res+1,r*res,(c-1)*res+1,c*res):copy(imgs[i])
    end
    return totalImg
end

------------------------------------------------------------------------------------------------------------

function colorHM(x)
    -- Converts a one-channel grayscale image to a color heatmap image
    local function gauss(x,a,b,c)
        return torch.exp(-torch.pow(torch.add(x,-b),2):div(2*c*c)):mul(a)
    end
    local cl = torch.zeros(3,x:size(1),x:size(2))
    cl[1] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
    cl[2] = gauss(x,1,.5,.3)
    cl[3] = gauss(x,1,.2,.3)
    cl[cl:gt(1)] = 1
    return cl
end

------------------------------------------------------------------------------------------------------------

function drawOutput(input, hms, coords)
    local im = drawSkeleton(input, hms, coords)

    local colorHms = {}
    local inp64 = image.scale(input,64):mul(.3)
    for i = 1,16 do
        colorHms[i] = colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    local totalHm = compileImages(colorHms, 4, 4, 64)
    im = compileImages({im,totalHm}, 1, 2, 256)
    im = image.scale(im,756)
    return im
end

function drawOutputFLIC(input, hms, coords)
    local im = drawSkeletonFLIC(input, hms, coords)

    local colorHms = {}
    local inp64 = image.scale(input,64):mul(.3)
    for i = 1,11 do
        colorHms[i] = colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    local totalHm = compileImages(colorHms, 4, 4, 64)
    im = compileImages({im,totalHm}, 1, 2, 256)
    --im = image.scale(im,756)
    return im
end

function drawHeatmapPartsFLIC(input, hms, coords)

    local colorHms = {}
    local inp64 = image.scale(input,64):mul(.3)
    for i = 1,11 do
        colorHms[i] = colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    return colorHms
end

function drawImgHeatmapParts(input, hms)
    local colorHms = {}
    local inp64 = image.scale(input,64):mul(.3)
    for i = 1,hms:size(1) do
        colorHms[i] = colorHM(hms[i])
        colorHms[i]:mul(.7):add(inp64)
    end
    return colorHms
end
