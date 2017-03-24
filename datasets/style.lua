--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local ImagenetDataset = torch.class('resnet.ImagenetDataset', M)

function ImagenetDataset:__init(imageInfo, opt, split)
   self.imageInfo = imageInfo[split]
   self.opt = opt
   self.split = split
   self.dir = paths.concat(opt.data, split)
   self.image_size = opt.image_size
   assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function ImagenetDataset:get(i)
  local path = ffi.string(self.imageInfo.imagePath[i]:data())

  local img = self:_loadImage(paths.concat(self.dir, path))
    
  if img:size(2) < self.image_size or img:size(3) < self.image_size  then 
    return nil 
  end
  
  return {
    img = img
  }
end


function ImagenetDataset:_loadImage(path)
   local ok, input = pcall(function()
      return image.load(path, 3, 'float')
   end)

   -- Sometimes image.load fails because the file extension does not match the
   -- image format. In that case, use image.decompress on a ByteTensor.
   if not ok then
      local f = io.open(path, 'r')
      assert(f, 'Error reading: ' .. tostring(path))
      local data = f:read('*a')
      f:close()

      local b = torch.ByteTensor(string.len(data))
      ffi.copy(b:data(), data, b:size(1))

      input = image.decompress(b, 3, 'float')
   end

   return input
end

function ImagenetDataset:size()
   if self.split == 'train' then
      return self.imageInfo.imageClass:size(1)
   else 
      return 1
   end
end

-- Computed from random subset of ImageNet training images
local meanstd = {
   mean = { 41.6 },
   std = { 23.9 },
}

function ImagenetDataset:preprocess()
   if self.split == 'train' then
      return t.Compose{
        t.RandomCrop(self.image_size),
      }
   elseif self.split == 'val' then
      return t.Compose{
        t.RandomCrop(self.image_size),
      }
   else
      error('invalid split: ' .. self.split)
   end
end

function ImagenetDataset:get_input_target()

  return function(img)
    return {
        input = img,
        target = img,
    }
  end
end

return M.ImagenetDataset
