--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split)
   local manualSeed = opt.manualSeed or 0
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      _G.get_input_target = dataset:get_input_target()

      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.nCrops = 1
   self.threads = threads
   self.__size = sizes[1][1]
   self.batchSize = opt.batch_size
end

function DataLoader:size()
   return math.ceil(self.__size / self.batchSize)
end

function DataLoader:get()
   self.loop = self.loop or self:run()
   
   local n, out = self.loop()

   if out then 
      return out
   else 
      print ('new loop')
      self.loop = self:run() 
      local n, out = self.loop()
      return out
   end
end

function DataLoader:run()
   local threads = self.threads
   local size, batchSize = self.__size, self.batchSize
   local perm = torch.randperm(size)

   local idx, sample = 1, nil
   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = torch.Tensor(batchSize):random(size)
         threads:addjob(
            function(indices, nCrops)
               
               local sz = indices:size(1)
               local batch_input, batch_target, imageSize
               
               for i, idx in ipairs(indices:totable()) do

                  -- if it's too small reject
                  local out = _G.dataset:get(idx)
                  
                  if not out then 
                     while true do
                        out = _G.dataset:get(torch.random(size))
                        if out then 
                           break
                        end
                     end
                  end

                  local img = _G.preprocess(out.img)
                  
                  local sample = _G.get_input_target(img)
                  local input = sample.input
                  local target = sample.target
        
                  if not batch_target then
                     imageSize = input:size():totable()
                     targetSize = target:size():totable()
                     -- if nCrops > 1 then table.remove(imageSize, 1) end
                     batch_input = torch.FloatTensor(sz, table.unpack(imageSize))
                     batch_target = torch.FloatTensor(sz, table.unpack(targetSize))
                  end
                  batch_input[i]:copy(sample.input)
                  batch_target[i]:copy(sample.target)

               end
               collectgarbage()
               
               return {
                  input = batch_input,
                  target = batch_target,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return -1, nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      -- local ss = sample.input:clone()
      return n, sample
   end

   return loop
end

return M.DataLoader
