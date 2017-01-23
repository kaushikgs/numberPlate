--usage: th loadData.lua dataset
require 'cutorch'
require 'math'
require 'image'
require 'nnx'      -- provides a normalization operator
require '../misc/mylib'

function countImages(path)
    local posCount = 0
    local negCount = 0
    local p = io.popen('ls -1 ' .. path .. 'positive/ | wc -l')
    posCount = posCount + tonumber(p:read(10)) --upto 10 digit numbers
    p:close()
    local p = io.popen('ls -1 ' .. path .. 'negative/ | wc -l')
    negCount = negCount + tonumber(p:read(10))
    p:close()
    return posCount, negCount
end

function loadImages(path, posCount, negCount)
    num = 1
    local images = torch.Tensor(posCount + negCount, 3, imheight, imwidth)
    local labels = torch.Tensor(posCount + negCount)
    --load from the dataset
    local p = io.popen('ls -p ' .. path .. 'positive/ | grep -v /')  --lists only files
    for file in p:lines() do
        images[num] = image.load(path .. 'positive/' .. file)
        labels[num] = 1
        num = num+1
    end
    p:close()
    print('Positive images loaded. Total ' .. num-1 .. ' images loaded')
    local p = io.popen('ls -p ' .. path .. 'negative/ | grep -v /')
    for file in p:lines() do
        images[num] = image.load(path .. 'negative/' .. file)
        labels[num] = 2
        num = num+1
    end
    p:close()
    print('Negative images loaded. Total ' .. num-1 .. ' images loaded')
    return images, labels
end

function loadData(dir)
    --count number of images first
    posCount, negCount = countImages(dir)
    --load the images
    images, labels = loadImages(dir, posCount, negCount)
    -- shuffle dataset: get shuffled indices in this variable:
    local shuffle = torch.randperm((#labels)[1])
    local numImages = posCount + negCount
    -- create dataset:
    data = {
       data = torch.Tensor(numImages, 3, imheight, imwidth),
       labels = torch.Tensor(numImages),
       size = function() return numImages end
    }
    for i=1,numImages do
       data.data[i] = images[shuffle[i]]:clone()
       data.labels[i] = labels[shuffle[i]]
    end
    -- remove from memory temp image files:
    images = nil
    labels = nil
    data.data = data.data:float()
    return data
end

-- classes: GLOBAL var!
classes = {'numberPlate','background'}

-- if #arg == 0 then
--     arg={...}
-- end
datName = ''

imwidth = 287
imheight = 95

local datasetDir = arg[1]
pathParts = split_path(datasetDir)
datasetName = pathParts[#pathParts]
datName = datasetDir .. datasetName .. '.dat'

trainDir = datasetDir .. 'train/'
valDir = datasetDir .. 'val/'

trainData = loadData(trainDir)
valData = loadData(valDir)

----------------------------------------------------------------------
print('==> preprocessing data')

local channels = {'r', 'g', 'b'}

local mean = {}
--local std = {}
for i,channel in ipairs(channels) do
    -- normalize each channel globally:
    mean[i] = trainData.data[{ {},i,{},{} }]:mean()
    --std[i] = trainData.data[{ {},i,{},{} }]:std()
    trainData.data[{ {},i,{},{} }]:add(-mean[i])
    --trainData.data[{ {},i,{},{} }]:div(std[i])
    valData.data[{ {},i,{},{} }]:add(-mean[i])   
    --valData.data[{ {},i,{},{} }]:div(std[i])
end

trainData.data = trainData.data:cuda()
trainData.labels = trainData.labels:cuda()
  
valData.data = valData.data:cuda()
valData.labels = valData.labels:cuda()

-- apart from return, save a table to file
dataset = {
    trainData = trainData,
    valData = valData,
    mean = mean,
    --std = std,
    classes = classes
}
torch.save(datName, dataset)
meanFile = datName:sub(1,-5) -- remove .dat
meanFile = meanFile .. '.mean'
meanFileObj = io.open(meanFile, 'w')
meanFileObj:write(mean[1] .. ' ' .. mean[2] .. ' ' .. mean[3]) -- .. '\n' .. std[1] .. ' ' .. std[2] .. ' ' .. std[3])
meanFileObj:close()
return dataset
