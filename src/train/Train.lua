require 'cunn'
require '../misc/mylib'
require 'optim'

------------------------------------ train function definition -----------------------------------
function nn.StochasticGradient:trainManual(trainData, valData, cnnfile, logFile)
    local iteration = 1
    local currentLearningRate = self.learningRate
    local module = self.module
    local criterion = self.criterion
    snapFreq = self.maxIteration/10
    logger = optim.Logger(logFile)
    logger:setNames{'# Epoch', '\t\tLoss', 'Tr. Accuracy', 'Val. Accuracy', 'time'}

    local shuffledIndices = torch.randperm(trainData:size(), 'torch.LongTensor')
    if not self.shuffleIndices then
        for t = 1,trainData:size() do
            shuffledIndices[t] = t
        end
    end

    local start_t = os.clock()
    --Calculate before starting training
    local preTrCorrect = 0
    for t = 1,trainData:size() do
        local example = trainData[t]
        local input = example[1]
        local target = example[2]

        module:forward(input)
     
        if module.output[1] >= module.output[2] and target == 1 then
            preTrCorrect = preTrCorrect+1
        else
            if module.output[1] < module.output[2] and target == 2 then
                preTrCorrect = preTrCorrect+1
            end
        end
    end
    local preTrAcc = preTrCorrect / trainData:size()

    local preValCorrect = 0
    for t = 1,valData:size() do
        local example = valData[t]
        local input = example[1]
        local target = example[2]

        module:forward(input)
     
        if module.output[1] >= module.output[2] and target == 1 then
            preValCorrect = preValCorrect+1
        else
            if module.output[1] < module.output[2] and target == 2 then
                preValCorrect = preValCorrect+1
            end
        end
    end
    local preValAcc = preValCorrect / valData:size()
   
    logger:add{0, 0, preTrAcc, preValAcc, os.clock()-start_t}

    while true do
        local currentError = 0
        local trCorrect = 0
        for t = 1,trainData:size() do
            local example = trainData[shuffledIndices[t]]
            local input = example[1]
            local target = example[2]

            currentError = currentError + criterion:forward(module:forward(input), target)

            module:updateGradInput(input, criterion:updateGradInput(module.output, target))
            module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)

            if module.output[1] >= module.output[2] and target == 1 then
                trCorrect = trCorrect+1
            else
                if module.output[1] < module.output[2] and target == 2 then
                    trCorrect = trCorrect+1
                end
            end

            if self.hookExample then
                self.hookExample(self, example)
            end
        end

        currentError = currentError / trainData:size()
        local trAcc = trCorrect / trainData:size()
 
        local valCorrect = 0
        for t = 1,valData:size() do
            local example = valData[t]
            local input = example[1]
            local target = example[2]

            module:forward(input)
         
            if module.output[1] >= module.output[2] and target == 1 then
                valCorrect = valCorrect+1
            else
                if module.output[1] < module.output[2] and target == 2 then
                    valCorrect = valCorrect+1
                end
            end
        end
        local valAcc = valCorrect / valData:size()

        if iteration%snapFreq == 0 then
            torch.save(cnnfile  .. '_' .. iteration .. '.cnn', self.module)
        end
      
        logger:add{iteration, currentError, trAcc, valAcc, os.clock()-start_t}

        if self.hookIteration then
            self.hookIteration(self, iteration, currentError)
        end

        iteration = iteration + 1
        currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
      
        if self.maxIteration > 0 and iteration > self.maxIteration then
            break
        end
    end
end

------------------------------------------ train function definition ----------------------------------
--usage: th Train.lua <epochs> <model file> <.dat file>


model = dofile(arg[2])
modeldirs = split_path(arg[2])
modelName = modeldirs[#modeldirs]
modelName = modelName:sub(1, modelName:len()-4)
dirsinpath = split_path(arg[3])
if arg[3]:ends('/') then    --a folder is passed as parameter
    assert(loadfile("loadData.lua"))(arg[3])
    datasetName = dirsinpath[#dirsinpath-1] --verify this
else                        --a dat file is passed
    dataset = torch.load(arg[3])
    datasetName = dirsinpath[#dirsinpath]
    datasetName = datasetName:sub(1, datasetName:len()-4) --remove .dat
end

setmetatable(dataset.trainData,     --make the data indexable
    {__index = function(t, i) 
                    return {t.data[i], t.labels[i]} 
                end}
);

setmetatable(dataset.valData, 
    {__index = function(t, i) 
                    return {t.data[i], t.labels[i]} 
                end}
);

trainer = nn.StochasticGradient(model.Model, model.loss)
trainer.learningRate = 0.001
trainer.maxIteration = tonumber(arg[1])
cnnFile = '../../trained_cnns/' .. datasetName .. '_' .. modelName
logFile = 'outputs/' .. datasetName .. '_' .. modelName .. '_' .. trainer.learningRate .. '.log'

trainer:trainManual(dataset.trainData, dataset.valData, cnnFile, logFile)
torch.save(cnnFile  .. '_' .. arg[1] .. '.cnn', model.Model)
print('Training completed. CNN saved as ' .. cnnFile)
