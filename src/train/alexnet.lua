require "cunn"

-- function makeDataParallel(model, nGPU)
--    if nGPU > 1 then
--       print('converting module to nn.DataParallelTable')
--       assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
--       local model_single = model
--       model = nn.DataParallelTable(1)
--       for i=1, nGPU do
--          cutorch.setDevice(i)
--          model:add(model_single:clone():cuda(), i)
--       end
--    end
--    cutorch.setDevice(opt.GPU)

--    return model
-- end

function createModel(nGPU)
   local model = nn.Sequential() -- branch 1
   model:add(nn.SpatialConvolution(3,96,11,11,4,4,2,2))       -- 287 -> 71, 95 -> 23
   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 71 -> 35, 23 -> 11
   model:add(nn.SpatialConvolution(96,256,5,5,1,1,2,2))       --  35 -> 35, 11 -> 11
   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(3,3,2,2))                   --  35 -> 17, 11 -> 5
   model:add(nn.SpatialConvolution(256,384,3,3,1,1,1,1))      --  17 -> 17, 5 -> 5
   model:add(nn.ReLU(true))
   model:add(nn.SpatialConvolution(384,384,3,3,1,1,1,1))      --  17 -> 17, 5 -> 5
   model:add(nn.ReLU(true))
   model:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  17 -> 17, 5 -> 5
   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 17 -> 8, 5 -> 2

   --features = makeDataParallel(features, nGPU) -- defined in util.lua

   -- 1.3. Create Classifier (fully connected layers)
   model:add(nn.View(256*8*2))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(256*8*2, 4096))
   model:add(nn.Threshold(0, 1e-6))
   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(4096, 4096))
   model:add(nn.Threshold(0, 1e-6))
   --model:add(nn.Linear(4096, nClasses))
   model:add(nn.Linear(4096, 2))
   model:add(nn.LogSoftMax())
   model:cuda()

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   --local model = nn.Sequential():add(features):add(classifier)
   --model.imageSize = 256
   --model.imageCrop = 224

   return model
end

local InputMaps = 3
local InputWidth = 287  --changed
local InputHeight = 95 --changed

model = createModel(1)
model = require('weight-init')(model, 'xavier')
for i,layer in ipairs(model.modules) do
    if layer.bias then
        layer.bias:fill(0)
    end
end

loss = nn.ClassNLLCriterion()
model = model:cuda()
loss = loss:cuda()

return {
    Model = model,
    loss = loss
}
