--[[
    Data sampling functions.
]]


-------------------------------------------------------------------------------
-- Data loading functions
-------------------------------------------------------------------------------

local function get_db_loader(name)
    local dbc = require 'dbcollection'

    local dbloader
    local str = string.lower(name)
    if str == 'flic' then
        dbloader = dbc.load{name='flic', task='keypoints', data_dir=opt.data_dir}
    elseif str == 'lsp' then
        dbloader = dbc.load{name='leeds_sports_pose_extended', task='keypoints', data_dir=opt.data_dir}
    elseif str == 'mpii' then
        dbloader = dbc.load{name='mpii_pose', task='keypoints', data_dir=opt.data_dir}
    elseif str == 'coco' then
        dbloader = dbc.load{name='coco', task='keypoints_2016', data_dir=opt.data_dir}
    else
        error(('Invalid dataset name: %s. Available datasets: mpii | flic | lsp | coco'):format(name))
    end
    return dbloader
end

------------------------------------------------------------------------------------------------------------

local function loader_flic(set_name)
    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str

    local dbloader = get_db_loader('flic')

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- number of keypoints (body joints)
    local nJoints = dbloader:size(set_name, 'keypoints')[2]

    -- data loader function
    local data_loader = function(idx)
        local data = dbloader:object(set_name, idx, true)

        local filename = paths.concat(dbloader.data_dir, ascii2str(data[1]))
        local keypoints = data[3]:float()
        local torso_bbox = data[2]:float():squeeze()
        local center = torch.FloatTensor({(torso_bbox[1]+torso_bbox[3])/2,
                                          (torso_bbox[2]+torso_bbox[4])/2})
        local scale = 2.2
        local normalize = torch.FloatTensor({torso_bbox[3]-torso_bbox[1],
                                             torso_bbox[4]-torso_bbox[2]}):norm()

        -- Load image
        local img = image.load(filename, 3, 'float')

        -- image, keypoints, center coords, scale, number of joints
        return img, keypoints:view(nJoints, 3), center, scale, nJoints, normalize
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_lsp(set_name)
    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str

    local dbloader = get_db_loader('lsp')

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- number of keypoints (body joints)
    local nJoints = dbloader:size(set_name, 'keypoints')[2]

    -- data loader function
    local data_loader = function(idx)
        local data = dbloader:object(set_name, idx, true)

        local filename = paths.concat(dbloader.data_dir, ascii2str(data[1]))
        local keypoints = data[2]:float():squeeze()
        local kps_coords_x = keypoints:select(2,1)
        local kps_coords_y = keypoints:select(2,2)
        local kps_x=kps_coords_x[kps_coords_x:gt(0)]
        local kps_y=kps_coords_y[kps_coords_y:gt(0)]
        local center = torch.FloatTensor({(kps_x:min() + kps_x:max()) / 2,
                                          (kps_y:min() + kps_y:max()) / 2})
        local scale = 1-- 1.25
        local normalize = (keypoints[10]-keypoints[3]):norm()

        -- Load image
        local img = image.load(filename, 3, 'float')

        -- image, keypoints, center coords, scale, number of joints
        return img, keypoints:view(nJoints, 3), center, scale, nJoints, normalize
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_mpii(set_name)
    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str

    local dbloader = get_db_loader('mpii')

    -- split train set annotations into two. (train + val)
    -- TODO

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- number of keypoints (body joints)
    local nJoints = dbloader:size(set_name, 'keypoints')[2]

    -- data loader function
    local data_loader = function(idx, flag_lsp)
        local data = dbloader:object(set_name, idx, true)

        local filename = paths.concat(dbloader.data_dir,  ascii2str(data[1]))
        local keypoints = data[5]:squeeze()
        local head_coord = data[4]:squeeze()
        local center = data[3]:squeeze()
        local scale = data[2]:squeeze()
        local normalize = torch.FloatTensor({head_coord[3]-head_coord[1],
                                             head_coord[4]-head_coord[2]}):norm() * 0.6

        -- Small adjustment so cropping is less likely to take feet out
        center[2] = center[2] --+ 15 * scale
        scale = scale --* 1.25

        -- Load image
        local img = image.load(filename, 3, 'float')

        -- image, keypoints, center coords, scale, number of joints
        if flag_lsp then
            -- This section is only used in conjunction with the lsp dataset
            local kps = keypoints:index(1,torch.LongTensor({1,2,3,4,5,6,11,12,13,14,15,16,9,10}))
            return img, kps, center, scale, kps:size(1), normalize
        else
            return img, keypoints:view(nJoints, 3), center, scale, nJoints, normalize
        end
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_coco(set_name)
    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str

    local dbloader = get_db_loader('coco')

    if set_name == 'test' then
        set_name = 'val' -- use coco val set for testing
    end

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- number of keypoints (body joints)
    local nJoints = dbloader:size(set_name, 'keypoints')[2]/3

    -- data loader function
    local data_loader = function(idx)
        local data = dbloader:object(set_name, idx, true)[1]

        local filename = paths.concat(dbloader.data_dir, ascii2str(data[1])[1])
        local num_keypoints = data[15][1]
        local keypoints = data[16][1]:view(nJoints, 3)

        -- discard annotations with less than 15 keypoints
        if num_keypoints < 12 then
            return {}
        end

        -- calc center coordinates
        local bbox = data[7][1]
        local center = torch.FloatTensor({(keypoints[6][1]+keypoints[13][1])/2,
                                          (keypoints[6][2]+keypoints[13][2])/2})
        --local center = torch.FloatTensor({(bbox[1]+bbox[3])/2, (bbox[2]+bbox[4])/2})
        local bbox_width = bbox[3]-bbox[1]
        local bbox_height = bbox[4]-bbox[2]
        local scale = math.max(bbox_height, bbox_width)/200 * 1.25
        local normalize = torch.FloatTensor({bbox_width, bbox_height}):norm() * 0.6

        -- Load image
        local img = image.load(filename, 3, 'float')

        -- image, keypoints, center coords, scale, number of joints
        return img, keypoints, center, scale, nJoints, normalize
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

--[[ Combine the LSP + MPII datasets ]]--
local function loader_lsp_mpii(set_name)
    local data_loader, set_size, nJoints
    if set_name == 'train' then
        local loader_mpii = loader_mpii(set_name)
        local loader_lsp = loader_lsp(set_name)

        -- number of samples per train/test sets
        local size_lsp = loader_lsp.size
        local size_mpii = loader_mpii.size
        set_size = size_lsp + size_mpii

        -- number of keypoints (body joints)
        nJoints = loader_lsp.num_keypoints

        data_loader = function(idx)
            if idx > size_lsp then
                return loader_mpii.loader(idx - size_lsp, true)
            else
                return loader_lsp.loader(idx)
            end
        end
    else
        local loader_lsp = loader_lsp(set_name)

        -- number of samples per train/test sets
        set_size = loader_lsp.size

        -- number of keypoints (body joints)
        nJoints = loader_lsp.num_keypoints

        data_loader = loader_lsp.loader
    end

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function fetch_loader_dataset(name, mode)
    local str = string.lower(name)
    if str == 'flic' then
        return loader_flic(mode)
    elseif str == 'lsp' then
        return loader_lsp(mode)
    elseif str == 'mpii' then
        return loader_mpii(mode)
    elseif str == 'coco' then
        return loader_coco(mode)
    elseif str == 'lsp+mpii' then
        return loader_lsp_mpii(mode)
    else
        error(('Invalid dataset name: %s. Available datasets: mpii | flic | lsp | coco | lsp+mpii.'):format(name))
    end
end

------------------------------------------------------------------------------------------------------------

function select_dataset_loader(name, mode)
    assert(name)
    assert(mode)

    local str = string.lower(mode)
    if str == 'train' then
        return {
            train = fetch_loader_dataset(name, 'train'),
            test = fetch_loader_dataset(name, 'test')
        }
    elseif str == 'test' then
        return {
            test = fetch_loader_dataset(name, 'test')
        }
    else
        error(('Invalid mode: %s. mode must be either \'train\' or \'test\''):format(mode))
    end
end