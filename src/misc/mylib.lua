require 'cutorch'
require 'cunn'
require 'image'

function split(str, pat)
    local t = {}  -- NOTE: use {n = 0} in Lua-5.0
    local fpat = "(.-)" .. pat
    local last_end = 1
    local s, e, cap = str:find(fpat, 1)
    while s do
       if s ~= 1 or cap ~= "" then
          table.insert(t,cap)
       end
       last_end = e+1
       s, e, cap = str:find(fpat, last_end)
    end
    if last_end <= #str then
       cap = str:sub(last_end)
       table.insert(t, cap)
    end
    return t
end

function split_path(str)
    return split(str,'[\\/]+')
end

function string.ends(String,End)
    return End=='' or string.sub(String,-string.len(End))==End
end

function validate(dataset, datasetName, cnn)
    trainData = dataset.trainData
    testData = dataset.testData

    correct = 0
    gt1 = 0
    p1 = 0
    pt1 = 0
    
    for i=1,trainData.size() do
        local groundtruth = trainData.labels[i]
        local prediction = cnn:forward(trainData.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        if groundtruth == indices[1] then
            correct = correct + 1
        end
        if indices[1] == 1 then
            pt1 = pt1 + 1
        end
        if groundtruth == 1 then
            gt1 = gt1+1
            if indices[1] == 1 then
                p1 = p1 + 1
            end
        end
    end
    print('Trainset accuracy: ', 100*correct/trainData.size() .. ' % ')
    print('Number plate precision: ', p1/pt1)
    print('Number plate recall: ', p1/gt1)

    correct = 0
    gt1 = 0
    p1 = 0
    pt1 = 0
    fp=0

    fpPath='results/'..datasetName..'/falsePositive/'
    tpPath='results/'..datasetName..'/truePositive/'
    fnPath='results/'..datasetName..'/falseNegative/'
    --io.popen('mkdir results results/'..datasetName.. ' '..fpPath  ..' '..tpPath  ..' '..fnPath )


    for i=1,testData.size() do
       -- normalize each channel globally:
        local groundtruth = testData.labels[i]
        local prediction = cnn:forward(testData.data[i])
        local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
        
         for j=1,3 do
	       --testData.data[{ i,j,{},{} }]:mul(dataset.std[j])
	       testData.data[{ i,j,{},{} }]:add(dataset.mean[j])
	    end
	
        if groundtruth == indices[1] then
            correct = correct + 1
        end
        if indices[1] == 1 then
        	--add to false positive
        	if groundtruth==2 then
        		--image.save(fpPath..fp..'.jpg',testData.data[i])
        		fp=fp+1
        	end
            pt1 = pt1 + 1
        end
        if groundtruth == 1 then
            gt1 = gt1+1
            if indices[1] == 1 then
            	--add to true positive
            	--image.save(tpPath..p1..'.jpg',testData.data[i])
                p1 = p1 + 1
            else
                --add to false negative
                --image.save(fnPath..gt1-p1..'.jpg',testData.data[i])
            end
        end
    end
    print('Testset accuracy: ', 100*correct/testData.size() .. ' % ')
    print('Number plate precision: ', p1/pt1)
    print('Number plate recall: ', p1/gt1)
end
