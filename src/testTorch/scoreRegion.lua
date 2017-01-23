require 'image'
require 'cunn'
require '../misc/mylib'
--math = require 'math'

cnn = torch.load(arg[1])
meanFilePath = arg[2] --mean file
imgPath = arg[3]

--cnn.modules[28] = nn.SoftMax()
--cnn:remove()
--cnn:add(nn.SoftMax():cuda())

meanFileObj = io.open(meanFilePath, "r")
meanLine = meanFileObj:read()
meanFileObj:close()
meanLineSplit = split(meanLine, " ")
mean = {}
mean[1] = tonumber(meanLineSplit[1])
mean[2] = tonumber(meanLineSplit[2])
mean[3] = tonumber(meanLineSplit[3])

channels = {'r', 'g', 'b'}

im = image.load(imgPath)
im = im:float()
im = im:cuda()
for i,channel in ipairs(channels) do
    im[{ i,{},{} }]:add(-mean[i])
end

torch.save('dbgTest.t7', im)

cnn:evaluate()

local prediction = cnn:forward(im)
--prediction = prediction:exp()
print('Score for ' .. imgPath .. ' is ' .. tostring(math.exp(prediction[1])))