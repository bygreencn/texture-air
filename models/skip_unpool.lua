-- Decoder-encoder like model with skip-connections

local act = function() return nn.LeakyReLU(nil, true) end

local nums_3x3down = {16, 32, 64, 128}
local nums_1x1 = {4, 4, 4, 4, 4}
local nums_3x3up = {16, 32, 64, 128}

local model = nn.Sequential()
if params.mode == 'texture' then
    net:add(nn.NoiseFill(3))
end

model_tmp = model

input_depth = 3
for i = 1,#nums_3x3down do
      
        local deeper = nn.Sequential()
        local skip = nn.Sequential()
        
        model_tmp:add(nn.Concat(2):add(skip):add(deeper))

        skip:add(conv(input_depth, nums_1x1[i], 1,1))
        skip:add(normalization(nums_1x1[i]))
        skip:add(act())
        
  
        local poolingModule = nn.SpatialMaxPooling(2,2,2,2)
        deeper:add(conv(input_depth, nums_3x3down[i], 3,1))
        deeper:add(normalization(nums_3x3down[i]))
        deeper:add(act())

        deeper:add(poolingModule)
             

        deeper:add(conv(nums_3x3down[i], nums_3x3down[i], 3))
        deeper:add(normalization(nums_3x3down[i]))
        deeper:add(act())

        deeper_main = nn.Sequential()

        if i == #nums_3x3down  then
            k = nums_3x3down[i]
        else
            deeper:add(deeper_main)
            
            deeper:add(conv(nums_3x3up[i+1], nums_3x3up[i], 3))
            deeper:add(normalization(nums_3x3down[i]))
            -- deeper:add(act())
            
            k = nums_3x3up[i]
        end

        -- deeper:add(nn.SpatialUpSamplingNearest(2))
        -- deeper:add(nn.Debug())
        deeper:add(nn.SpatialMaxUnpooling(poolingModule))
        -- deeper:add(nn.Copy())

        deeper:add(normalization(k))
        skip:add(normalization(nums_1x1[i]))
        

        model_tmp:add(conv(nums_1x1[i] +  k, nums_3x3up[i], 3))
        model_tmp:add(normalization(nums_3x3up[i]))
        model_tmp:add(act())

        model_tmp:add(conv(nums_3x3up[i], nums_3x3up[i], 3))
        model_tmp:add(normalization(nums_3x3up[i]))
        model_tmp:add(act())

        model_tmp:add(conv(nums_3x3up[i], nums_3x3up[i], 1))
        model_tmp:add(normalization(nums_3x3up[i]))
        model_tmp:add(act())
        
        
        input_depth = nums_3x3down[i]
        model_tmp = deeper_main
end
model:add(conv(nums_3x3up[1], 3, 1,1))
model = nn.Sequential():add(nn.ConcatTable():add(nn.Identity()):add(model)):add(nn.CAddTable())


return model:add(nn.TVLoss(params.tv_weight))