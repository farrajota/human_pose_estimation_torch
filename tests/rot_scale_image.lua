--[[
    Test the data generator.
--]]

require 'torch'
require 'paths'
require 'string'
require 'image'
disp = require 'display'

torch.manualSeed(41)

paths.dofile('../projectdir.lua') -- Project directory
utils = paths.dofile('../util/utils.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'lsp'

paths.dofile('../dataset.lua')
paths.dofile('../data.lua')


dataset = loadDataset() -- load dataset train+val+test set

idx = 3500
mode = 'train'
local img, keypoints, c, s, nJoints = loadImgKeypointsFn_(dataset[mode], idx)  

s_factor = torch.uniform(1-opt.scale,1+opt.scale)
--s_factor =0.75
print('scaling factor: ' .. s_factor)
s = s * s_factor

-- apply rescale image
local rescale_factor = (200*s)/opt.inputRes
local ht,wd = img:size(2), img:size(3)
newSize = math.floor(math.max(ht,wd) / rescale_factor)
c_ = c:float()/rescale_factor
img_scaled = image.scale(img, newSize)

--print(#img)
--print(#img_scaled)

--disp.image(image.drawRect(img,c[1]-1,c[2]-1,c[1]+1,c[2]+1,{lineWidth = 3, color = {0, 255, 0}}))
--disp.image(image.drawRect(img_scaled,c_[1]-1,c_[2]-1,c_[1]+1,c_[2]+1,{lineWidth = 3, color = {0, 255, 0}}))

dispImg = image.drawRect(img,c[1]-1,c[2]-1,c[1]+1,c[2]+1,{lineWidth = 3, color = {0, 255, 0}})
for i=1, keypoints:size(1) do
    if keypoints[i][1] > 0 and keypoints[i][2] > 0 then
        dispImg = image.drawRect(dispImg,keypoints[i][1]-1,keypoints[i][2]-1,keypoints[i][1]+1,keypoints[i][2]+1,{lineWidth = 3, color = {0, 255, 0}})
    end
end
disp.image(dispImg,{title='Original'})

-- copy image array to new array (if rotation is performed)
local rot = torch.uniform(-opt.rotate,opt.rotate)
--rot = 0
print('rotation: ' .. rot)
c_=c_:int()
--[[
local out
if rot == 0 then
    out = torch.FloatTensor(3, opt.inputRes, opt.inputRes):fill(0)
    -- copy data to the output tensor
    local x1,y1 = math.max(1,c_[1]-128),math.max(1,c_[2]-128)
    local x2,y2 = math.min(img_scaled:size(3),c_[1]+127),math.min(img_scaled:size(2),c_[2]+127)
    local x1_,y1_ = 128-(c_[1]-x1)+1, 128-(c_[2]-y1)+1
    local x2_,y2_ = x1_+(x2-x1), y1_+(y2-y1)
    out[{{},{y1_,y2_},{x1_,x2_}}]:copy(img_scaled[{{},{y1,y2},{x1,x2}}])
else
    offset = opt.inputRes/2
    out = torch.FloatTensor(3, opt.inputRes+offset, opt.inputRes+offset):fill(0)
    -- copy data to the output tensor
    offset_left, offset_top = (opt.inputRes+offset)/2, (opt.inputRes+offset)/2
    offset_right, offset_bot = (opt.inputRes+offset)/2-1, (opt.inputRes+offset)/2-1
    local x1,y1 = math.max(1,c_[1]-offset_left),math.max(1,c_[2]-offset_top)
    local x2,y2 = math.min(img_scaled:size(3),c_[1]+offset_right),math.min(img_scaled:size(2),c_[2]+offset_bot)
    local x1_,y1_ = offset_left-(c_[1]-x1)+1, offset_top-(c_[2]-y1)+1
    local x2_,y2_ = x1_+(x2-x1), y1_+(y2-y1)
    out[{{},{y1_,y2_},{x1_,x2_}}]:copy(img_scaled[{{},{y1,y2},{x1,x2}}])
    
    -- apply rotation 
    out_rot = image.rotate(out, rot * math.pi / 180, 'bilinear')
    
    disp.image(out_rot)
    
    out_crop = image.crop(out_rot,'c',opt.inputRes,opt.inputRes)
    
    disp.image(out_crop)
end

disp.image(out)
--]]

-- apply scalling
offset = opt.inputRes/2
out = torch.FloatTensor(3, opt.inputRes+offset, opt.inputRes+offset):fill(0)
-- copy data to the output tensor
offset_left, offset_top = (opt.inputRes+offset)/2, (opt.inputRes+offset)/2
offset_right, offset_bot = (opt.inputRes+offset)/2-1, (opt.inputRes+offset)/2-1
local x1,y1 = math.max(1,c_[1]-offset_left),math.max(1,c_[2]-offset_top)
local x2,y2 = math.min(img_scaled:size(3),c_[1]+offset_right),math.min(img_scaled:size(2),c_[2]+offset_bot)
local x1_,y1_ = offset_left-(c_[1]-x1)+1, offset_top-(c_[2]-y1)+1
local x2_,y2_ = x1_+(x2-x1), y1_+(y2-y1)
out[{{},{y1_,y2_},{x1_,x2_}}]:copy(img_scaled[{{},{y1,y2},{x1,x2}}])
--disp.image(out,{title='After scalling'})

-- apply rotation
if rot ~= 0 then
    out = image.rotate(out, rot * math.pi / 180, 'bilinear')
    --disp.image(out,{title='After rotation'})
end

-- apply cropping
out = image.crop(out,'c',opt.inputRes,opt.inputRes)
--disp.image(out,{title='After cropping'})



------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------

function getTransformV2(center, scale, rot, res)
    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

--[[
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
        t_[1][3] = res/2
        t_[2][3] = res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end
    --]]
    
    -- Rotation
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        t[1][1] = c
        t[1][2] = -s
        t[2][1] = s
        t[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = res/2
        t_[2][3] = res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    return t
end


function getTransformV4(center, scale, rot, res)
    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

--
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
        t_[1][3] = res/2
        t_[2][3] = res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end
    --

    return t
end


function transformV2(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1

    local t = getTransformV2(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2)

    return new_point:add(1)
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
       -- t =  r *t
      --
      -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
        --]]
    end
    
    -- comb transforms
    local c_matrix = t
    if invert then
        c_matrix = torch.inverse(c_matrix)
    end
    
    -- process transformation
    local new_point = (c_matrix*pt_):sub(1,2)
    
    return new_point
end

function mytransformV2(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1
    
    local h = 200 * scale
    local t_matrix = torch.eye(3)

    -- Scaling
    t_matrix[1][1] = res / h
    t_matrix[2][2] = res / h

    -- Translation
    t_matrix[1][3] = res * (-center[1] / h + .5)
    t_matrix[2][3] = res * (-center[2] / h + .5)
    
    -- comb transforms
    local c_matrix = t_matrix
    if invert then
        c_matrix = torch.inverse(c_matrix)
    end
    
    -- process transformation
    local new_point = (c_matrix*pt_):sub(1,2)
    
    return new_point:add(1)
end

function mytransformV3(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1],pt[2]
    
    local h = 200 * scale
    local t_matrix = torch.eye(3)
    
    -- Translation
    t_matrix[1][3] = (-center[1] )
    t_matrix[2][3] = (-center[2] )
    
    -- rotation matrix
    local r_matrix = torch.eye(3)
    if rot ~= 0 then
        local rotation = -rot
        local r = torch.eye(3)
        local ang = rotation * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r_matrix[1][1] = c
        r_matrix[1][2] = -s
        r_matrix[2][1] = s
        r_matrix[2][2] = c
        -- Need to make sure rotation is around center
 --       local t_ = torch.eye(3)
 --       t_[1][3] = res/2
 --       t_[2][3] = res/2
 --       local t_inv = torch.eye(3)
 --       t_inv[1][3] = res/4
 --       t_inv[2][3] = res/4
 --       r_matrix = r_matrix * t_ 
    end
    
    -- comb transforms
    local c_matrix = t_matrix*r_matrix*torch.inverse(t_matrix)
    if invert then
        c_matrix = torch.inverse(c_matrix)
    end
    
    -- process transformation
    local new_point = (c_matrix*pt_):sub(1,2)
    
    return new_point:add(1)
end

function mytransformV4(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1],pt[2]
    
    -- comb transforms
    local c_matrix = getTransformV4(center, scale, rot, res)
    if invert then
        c_matrix = torch.inverse(c_matrix)
    end
    
    -- process transformation
    local new_point = (c_matrix*pt_):sub(1,2)
    
    return new_point:add(1)
end



new_kps = torch.FloatTensor(14,2):fill(0)
for i=1, new_kps:size(1) do
    if keypoints[i][1]>0 and keypoints[i][2]>0 then
        new_kps[i] = mytransform(keypoints[i], c, s, rot, opt.inputRes, false) 
        --new_kps[i] = mytransformV4(keypoints[i], c, s, rot, opt.inputRes, false) 
        
        --new_kps[i] = mytransformV3(new_kps[i], torch.FloatTensor(2):fill(128), 0, rot, opt.inputRes, false) 
        aqui=1
        --new_kps[i] = transformV2(new_kps[i], torch.FloatTensor(2):fill(127), 1, rot, opt.inputRes, false) 
    end
end


cropImg = out:clone()--image.drawRect(out,c_[1]-1,c_[2]-1,c_[1]+1,c_[2]+1,{lineWidth = 3, color = {0, 255, 0}})
for i=1, keypoints:size(1) do
    if new_kps[i][1] > 0 and new_kps[i][2] > 0 then
        cropImg = image.drawRect(cropImg,new_kps[i][1]-1,new_kps[i][2]-1,new_kps[i][1]+1,new_kps[i][2]+1,{lineWidth = 3, color = {0, 255, 0}})
    end
end
disp.image(cropImg,{title='After cropping + keypoints'})

-- undo rotation
unrotateImg = out:clone()
unrotateImg = image.rotate(unrotateImg, -rot * math.pi / 180, 'bilinear')
unrot_kps = torch.FloatTensor(14,2):fill(0)
for i=1, unrot_kps:size(1) do
    if keypoints[i][1]>0 and keypoints[i][2]>0 then
        unrot_kps[i] = mytransform(new_kps[i], c, s, rot, opt.inputRes, true) 
        --new_kps[i] = mytransformV4(keypoints[i], c, s, rot, opt.inputRes, false) 
        
        --new_kps[i] = mytransformV3(new_kps[i], torch.FloatTensor(2):fill(128), 0, rot, opt.inputRes, false) 
        aqui=1
        --new_kps[i] = transformV2(new_kps[i], torch.FloatTensor(2):fill(127), 1, rot, opt.inputRes, false) 
    end
end

print('Original Keypoints')
print(keypoints)
print('transformed keypoints')
print(new_kps)
print('inverse transform keypoints')
print(unrot_kps)

for i=1, keypoints:size(1) do
    if unrot_kps[i][1] > 0 and unrot_kps[i][2] > 0 then
        unrotateImg = image.drawRect(unrotateImg,unrot_kps[i][1]-1,unrot_kps[i][2]-1,unrot_kps[i][1]+1,unrot_kps[i][2]+1,{lineWidth = 3, color = {0, 255, 0}})
    end
end
disp.image(unrotateImg,{title='Rotation removed'})


print(#out)