-------------------------------------------------------------------------------
-- Coordinate normalization
-------------------------------------------------------------------------------

function normCoords(bbox, keypoints, size, scale)

    -- get crop region coordinates
    local xc = (bbox[1]+bbox[3])/2
    local yc = (bbox[2]+bbox[4])/2
    local width = (bbox[3]-bbox[1])+1
    local height = (bbox[5]-bbox[2])+1

    -- get maximum size of the bbox
    local max_size = math.max(width, height)

    -- get a square bbox around the center of the bbox and add a small scaling effect
    local crop_coords
    if width > height then
        local diff = (width - height)/2
        local w_offset = (width/2) * scale
        local h_offset = (height/2+diff) * scale
        crop_coords = {xc-w_offset,yc-h_offset,xc+w_offset,yc+h_offset}
    else
        local diff = (height - width)/2
        local w_offset = (width/2+diff) * scale
        local h_offset = (height/2) * scale
        crop_coords = {xc-w_offset,yc-h_offset,xc+w_offset,yc+h_offset}
    end

    -- set new keypoints coordinates
    local new_keypoints = keypoints:clone():fill(0)
    local new_max_size = crop_coords[3]-crop_coords[1]
    for i=1, keypoints:size(1) do
        keypoints[i][1] = keypoints[i][1] - crop_coords[1] / new_max_size * size
        keypoints[i][2] = keypoints[i][2] - crop_coords[2] / new_max_size * size
    end

    -- output
    return crop_coords, new_keypoints
end


-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------

function getTransform(center, scale, rot, res)
    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- Rotation
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    return t
end

function transform(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1

    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2)

    return new_point:int():add(1)
end

function transformPreds(coords, center, scale, res)
    local origDims = coords:size()
    coords = coords:view(-1,2)
    local newCoords = coords:clone()
    for i = 1,coords:size(1) do
        newCoords[i] = transform(coords[i], center, scale, 0, res, 1)
    end
    return newCoords:view(origDims)
end

function mytransform(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1],pt[2]

    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- rotation matrix
    if rot ~= 0 then
        local rotation = -rot
        local r = torch.eye(3)
        local ang = rotation * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    if invert then
        t = torch.inverse(t)
    end

    -- process transformation
    local new_point = (t*pt_):sub(1,2)

    return new_point
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------

function checkDims(dims)
    return dims[3] < dims[4] and dims[5] < dims[6]
end

function crop2(img, center, scale, rot, res)
    local ndim = img:nDimension()
    if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end
    local ht,wd = img:size(2), img:size(3)
    local tmpImg,newImg = img, torch.zeros(img:size(1), res, res)

    -- Modify crop approach depending on whether we zoom in/out
    -- This is for efficiency in extreme scaling cases
    local scaleFactor = (200 * scale) / res
    if scaleFactor < 2 then
        scaleFactor = 1
    else
        local newSize = math.floor(math.max(ht,wd) / scaleFactor)
        if newSize < 2 then
            -- Zoomed out so much that the image is now a single pixel or less
            if ndim == 2 then
                newImg = newImg:view(newImg:size(2),newImg:size(3))
            end
            return newImg
        else
           tmpImg = image.scale(img,newSize)
           ht,wd = tmpImg:size(2),tmpImg:size(3)
        end
    end

    -- Calculate upper left and bottom right coordinates defining crop region
    local c,s = center:float()/scaleFactor, scale/scaleFactor
    local ul = transform({1,1}, c, s, 0, res, true)
    local br = transform({res+1,res+1}, c, s, 0, res, true)
    if scaleFactor >= 2 then br:add(-(br - ul - res)) end

    -- If the image is to be rotated, pad the cropped area
    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then ul:add(-pad); br:add(pad) end

    -- Define the range of pixels to take from the old image
    local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
                       math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
    -- And where to put them in the new image
    local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
                       math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}

    -- Initialize new image and copy pixels over
    local newImg = torch.zeros(img:size(1), br[2] - ul[2], br[1] - ul[1])
    if not pcall(function() newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_))) end) then
       print("Error occurred during crop!")
    end

    if rot ~= 0 then
        -- Rotate the image and remove padded area
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        newImg = newImg:sub(1,-1,pad+1,newImg:size(2)-pad,pad+1,newImg:size(3)-pad):clone()
    end

    if scaleFactor < 2 then newImg = image.scale(newImg,res,res) end
    if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
    return newImg
end

function twoPointCrop(img, s, pt1, pt2, pad, res)
    local center = (pt1 + pt2) / 2
    local scale = math.max(20*s,torch.norm(pt1 - pt2)) * .007
    scale = scale * pad
    local angle = math.atan2(pt2[2]-pt1[2],pt2[1]-pt1[1]) * 180 / math.pi - 90
    return crop(img, center, scale, angle, res)
end

function mycrop(img, center, scale, rot, res)

    -- setup parameters
    local rescale_factor = (200*scale)/res
    local ht,wd = img:size(2), img:size(3)
    local newSize = math.floor(math.max(ht,wd) / rescale_factor)
    local c_ = center:float()/rescale_factor
    c_=c_:int()

    -- apply scalling
    local img_scaled = image.scale(img, newSize)
    local offset = res/2
    local out = torch.FloatTensor(3, res+offset, res+offset):fill(0)
    local offset_left, offset_top = (res+offset)/2, (res+offset)/2
    local offset_right, offset_bot = (res+offset)/2-1, (res+offset)/2-1
    -- (set copy limits)
    local x1,y1 = math.max(1,c_[1]-offset_left),math.max(1,c_[2]-offset_top)
    local x2,y2 = math.min(img_scaled:size(3),c_[1]+offset_right),math.min(img_scaled:size(2),c_[2]+offset_bot)
    local x1_,y1_ = offset_left-(c_[1]-x1)+1, offset_top-(c_[2]-y1)+1
    local x2_,y2_ = x1_+(x2-x1), y1_+(y2-y1)
    -- (copy data to the output tensor)
    assert(y1_>0 and y2_>0 and x1_>0 and x2_>0, 'One of this variables is smaller than 1: x1_,y1_,x2_,y2_')
    out[{{},{y1_,y2_},{x1_,x2_}}]:copy(img_scaled[{{},{y1,y2},{x1,x2}}])

    -- apply rotation
    if rot ~= 0 then
        out = image.rotate(out, rot * math.pi / 180, 'bilinear')
    end

    -- apply cropping (center)
    out = image.crop(out,'c',res,res)

    return out
end


-------------------------------------------------------------------------------
-- Non-maximum Suppression
-------------------------------------------------------------------------------

function localMaxes(hm, n, c, s, hmIdx, nmsWindowSize)
    -- Set up max network for NMS
    local nmsWindowSize = nmsWindowSize or 3
    local nmsPad = (nmsWindowSize - 1)/2
    local maxlayer = nn.Sequential()
    if cudnn then
        maxlayer:add(cudnn.SpatialMaxPooling(nmsWindowSize, nmsWindowSize,1,1, nmsPad, nmsPad))
        maxlayer:cuda()
    else
        maxlayer:add(nn.SpatialMaxPooling(nmsWindowSize, nmsWindowSize,1,1, nmsPad,nmsPad))
        maxlayer:float()
    end
    maxlayer:evaluate()

    local hmSize = torch.totable(hm:size())
    hm = torch.Tensor(1,unpack(hmSize)):copy(hm):float()
    if hmIdx then hm = hm:sub(1,-1,hmIdx,hmIdx) end
    local hmDim = hm:size()
    local max_out
    -- First do nms
    if cudnn then
        max_out = maxlayer:forward(hm:cuda())
        cutorch.synchronize()
    else
        max_out = maxlayer:forward(hm)
    end

    local nms = torch.cmul(hm, torch.eq(hm, max_out:float()):float())[1]
    -- Loop through each heatmap retrieving top n locations, and their scores
    local predCoords = torch.Tensor(hmDim[2], n, 2)
    local predScores = torch.Tensor(hmDim[2], n)
    for i = 1, hmDim[2] do
        local nms_flat = nms[i]:view(nms[i]:nElement())
        local vals,idxs = torch.sort(nms_flat,1,true)
        for j = 1,n do
            local pt = {(idxs[j]-1) % hmSize[3] + 1, math.floor((idxs[j]-1) / hmSize[3]) + 1 }
            if c then
                predCoords[i][j] = transform(pt, c, s, 0, hmSize[#hmSize], true)
            else
                predCoords[i][j] = torch.Tensor(pt)
            end
            predScores[i][j] = vals[j]
        end
    end
    return predCoords, predScores
end

-------------------------------------------------------------------------------
-- Draw gaussian
-------------------------------------------------------------------------------

function drawGaussian(img, pt, sigma)
    -- Draw a 2D gaussian
    -- Check that any part of the gaussian is in-bounds
    local tmpSize = math.ceil(3*sigma)
    local ul = {math.floor(pt[1] - tmpSize), math.floor(pt[2] - tmpSize)}
    local br = {math.floor(pt[1] + tmpSize), math.floor(pt[2] + tmpSize)}
    -- If not, return the image as is
    if (ul[1] > img:size(2) or ul[2] > img:size(1) or br[1] < 1 or br[2] < 1) then return img end
    -- Generate gaussian
    local size = 2*tmpSize + 1
    local g = image.gaussian(size)
    -- Usable gaussian range
    local g_x = {math.max(1, -ul[1]), math.min(br[1], img:size(2)) - math.max(1, ul[1]) + math.max(1, -ul[1])}
    local g_y = {math.max(1, -ul[2]), math.min(br[2], img:size(1)) - math.max(1, ul[2]) + math.max(1, -ul[2])}
    -- Image range
    local img_x = {math.max(1, ul[1]), math.min(br[1], img:size(2))}
    local img_y = {math.max(1, ul[2]), math.min(br[2], img:size(1))}
    assert(g_x[1] > 0 and g_y[1] > 0)
    img:sub(img_y[1], img_y[2], img_x[1], img_x[2]):cmax(g:sub(g_y[1], g_y[2], g_x[1], g_x[2]))
    return img
end

function drawLine(img, pt1, pt2, width, color)
    if img:nDimension() == 2 then img = img:view(1,img:size(1),img:size(2)) end
    local nChannels = img:size(1)
    color = color or torch.ones(nChannels)
    if type(pt1) == 'table' then pt1 = torch.Tensor(pt1) end
    if type(pt2) == 'table' then pt2 = torch.Tensor(pt2) end

    m = pt1:dist(pt2)
    dy = (pt2[2] - pt1[2])/m
    dx = (pt2[1] - pt1[1])/m
    for j = 1,width do
        start_pt1 = torch.Tensor({pt1[1] + (-width/2 + j-1)*dy, pt1[2] - (-width/2 + j-1)*dx})
        start_pt1:ceil()
        for i = 1,torch.ceil(m) do
            y_idx = torch.ceil(start_pt1[2]+dy*i)
            x_idx = torch.ceil(start_pt1[1]+dx*i)
            if y_idx - 1 > 0 and x_idx -1 > 0
            and y_idx < img:size(2) and x_idx < img:size(3) then
                for j = 1,nChannels do img[j]:sub(y_idx-1,y_idx,x_idx-1,x_idx):fill(color[j]) end
            end
        end
    end
end

-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------

function shuffleLR(x)
    local dim
    if x:nDimension() == 4 then
        dim = 2
    else
        assert(x:nDimension() == 3)
        dim = 1
    end

    local matchedParts
    if opt.dataset == 'mpii' then
        matchedParts = {
            {1,6},   {2,5},   {3,4},
            {11,16}, {12,15}, {13,14}
        }
    elseif opt.dataset == 'flic' then
        matchedParts = {
            {1,4}, {2,5}, {3,6}, {7,8}, {9,10}
        }
    elseif opt.dataset == 'lsp' or opt.dataset =='mpii+lsp' then
        matchedParts = {
            {1,6}, {2,5}, {3,4}, {7,12}, {8,11}, {9,10}
        }
    elseif opt.dataset == 'coco'  then
        matchedParts = {
            {2,3}, {4,5}, {6,7}, {8,9}, {10,11}, {12,13}, {14,15}, {16,17}
        }
    end

    for i = 1,#matchedParts do
        local idx1, idx2 = unpack(matchedParts[i])
        local tmp = x:narrow(dim, idx1, 1):clone()
        x:narrow(dim, idx1, 1):copy(x:narrow(dim, idx2, 1))
        x:narrow(dim, idx2, 1):copy(tmp)
    end

    return x
end

function flip(x)
    require 'image'
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end
