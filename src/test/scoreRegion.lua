require 'image'
require 'cunn'
require '../misc/mylib'
--math = require 'math'

cnn = torch.load(arg[1])
meanFilePath = arg[2] --mean file
imgPath = arg[3]

meanFileObj = io.open(meanFilePath, "r")
meanLine = meanFileObj:read()
meanFileObj:close()
meanLineSplit = split(meanLine, " ")
mean = {}
mean[1] = tonumber(meanLineSplit[1])
mean[2] = tonumber(meanLineSplit[2])
mean[3] = tonumber(meanLineSplit[3])

imwidth = 150
imheight = 150

channels = {'r', 'g', 'b'}

im = image.load(imgPath)
im = im:float()
im = im:cuda()
for i,channel in ipairs(channels) do
    im[{ i,{},{} }]:add(-mean[i])
end

cnn:evaluate()

local prediction = cnn:forward(im)
print('Score for ' .. imgPath .. ' is ' .. tostring(prediction[1]))
