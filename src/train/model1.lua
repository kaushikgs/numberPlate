require 'cunn'

local opt = opt or {type = 'cuda', net='new'}

if opt.type == 'cuda' then
    require 'cunn'
end

local InputMaps = 3
local InputWidth = 300  --changed
local InputHeight = 100 --changed

local KernelSize = {7,7,3,3,3,3,1,1,1,1}
local ConvStride = {2,1,1,1,1,1,1,1,1,1}
local Padding = {0,0,1,1,1,1,0,0,0,0}
local PoolSize =   {3,2,1,1,1,3,1,1,1,1}
local PoolStride= PoolSize
local TValue = 0
local TReplace = 0
local Outputs = 2 --changed
--local FeatMaps = {InputMaps, 96,256,384,384,256,256,4096,4096, Outputs}
local FeatMaps = {InputMaps, 96,256,512,512,1024,1024,4096, 64, Outputs} --changed

local LayerNum

--------------Calculate size of feature maps - useful for linear layer flattening------------------------
SizeMapWidth = {InputWidth}
for i=2, #FeatMaps do
    SizeMapWidth[i] = math.floor(math.ceil((SizeMapWidth[i-1] - KernelSize[i-1] + 1 + 2*Padding[i-1]) / ConvStride[i-1]) / PoolStride[i-1])
end
SizeMapHeight = {InputHeight}
for i=2, #FeatMaps do
    SizeMapHeight[i] = math.floor(math.ceil((SizeMapHeight[i-1] - KernelSize[i-1] + 1 + 2*Padding[i-1]) / ConvStride[i-1]) / PoolStride[i-1])
end

----------------Create Model-------------------------------------
model = nn.Sequential()

---------------Layer - Convolution + Max Pooling------------------
LayerNum = 1
model:add(nn.SpatialConvolutionMM(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(PoolSize[LayerNum], PoolSize[LayerNum], PoolStride[LayerNum], PoolStride[LayerNum]))


---------------Layer - Convolution + Max Pooling------------------
LayerNum = 2
model:add(nn.SpatialConvolutionMM(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(PoolSize[LayerNum], PoolSize[LayerNum], PoolStride[LayerNum], PoolStride[LayerNum]))

---------------Layer - Convolution ------------------
LayerNum = 3
model:add(nn.SpatialZeroPadding(1,1,1,1))
model:add(nn.SpatialConvolutionMM(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.ReLU())


---------------layer - convolution ------------------
LayerNum = 4
model:add(nn.SpatialZeroPadding(1,1,1,1))
model:add(nn.SpatialConvolutionMM(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.ReLU())


---------------layer - convolution ------------------
LayerNum = 5
model:add(nn.SpatialZeroPadding(1,1,1,1))
model:add(nn.SpatialConvolutionMM(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.ReLU())


---------------Layer - Convolution + Max Pooling------------------
LayerNum = 6
model:add(nn.SpatialZeroPadding(1,1,1,1))
model:add(nn.SpatialConvolutionMM(FeatMaps[LayerNum], FeatMaps[LayerNum+1], KernelSize[LayerNum], KernelSize[LayerNum], ConvStride[LayerNum], ConvStride[LayerNum]))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(PoolSize[LayerNum], PoolSize[LayerNum], PoolStride[LayerNum], PoolStride[LayerNum]))

---------------Layer - Fully connected ------------------
LayerNum = 7
model:add(nn.Reshape(SizeMapWidth[LayerNum]*SizeMapHeight[LayerNum]*FeatMaps[LayerNum]))
model:add(nn.Linear(SizeMapWidth[LayerNum]*SizeMapHeight[LayerNum]*FeatMaps[LayerNum],  FeatMaps[LayerNum+1]))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
---------------Layer - Fully connected ------------------
LayerNum = 8
model:add(nn.Linear(FeatMaps[LayerNum], FeatMaps[LayerNum+1]))
model:add(nn.ReLU())
model:add(nn.Dropout(0.5))
---------------Layer - Fully connected classifier ------------------
LayerNum = 9
model:add(nn.Linear(FeatMaps[LayerNum], FeatMaps[LayerNum+1]))


---------------Layer - Log Probabilities--------------------------
model:add(nn.LogSoftMax())

model = require('weight-init')(model, 'xavier_caffe')
for i,layer in ipairs(model.modules) do
    if layer.bias then
        layer.bias:fill(0)
    end
end

--
--if (opt.net ~= 'new') then
--    print '==> Loaded Net'
--    model = torch.load(opt.net);
--    model = model:cuda()
--
--
--else
--    print '==> New Net'
--    -- adjust all biases for threshold activation units
--    local finput = model.modules[1].finput
--    local fgradInput = model.modules[1].fgradInput
--    for i,layer in ipairs(model.modules) do
--        if layer.bias then
--            layer.bias:fill(.01)
--        end
--        if layer.finput then
--            layer.finput = finput
--        end
--        if layer.fgradInput then
--            layer.fgradInput = fgradInput
--        end
--    end
--
--end
--

local w,dE_dw = model:getParameters()
-- w:copy(torch.load('weights'))
---- Loss: NLL
loss = nn.ClassNLLCriterion()--passed weight as parameter(13/07/16)
----------------------------------------------------------------------

if opt.type == 'cuda' then
    model:cuda()
    loss:cuda()
end

return {
    Model = model,
    Weights = w,
    Grads = dE_dw,
    FeatMaps = FeatMaps,
    SizeMap = SizeMap,
    loss = loss
}

