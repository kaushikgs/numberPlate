require 'cunn'

function processLayers(cnn, layers, tempFile, layerPrefix)
    for i,layerno in ipairs(layers) do
        weights = cnn.modules[layerno].weight
        biases = cnn.modules[layerno].bias
        if layerPrefix=='ip' then
            tempFile:write(layerPrefix .. tostring(i+6) .. ' ' .. weights:size(1) .. ' ' .. weights:size(2) .. '\n')
        else
            tempFile:write(layerPrefix .. tostring(i) .. ' ' .. weights:size(1) .. ' ' .. weights:size(2) .. '\n')
        end
        
        for i=1,weights:size(1) do --iterate over rows
            line = ''
            for j=1,weights:size(2) do
                line = line .. weights[i][j] .. ' '
            end
            tempFile:write(line .. '\n')
        end
        
        line = ''
        for i=1,biases:size(1) do
            line = line .. biases[i] .. ' '
        end
        tempFile:write(line .. '\n')
    end
end

tempFileName = arg[2]
convolutions = {1,4,8,11,14,17}
full = {21,24,27}

cnn = torch.load(arg[1])
tempFile = io.open(tempFileName, 'w')
processLayers(cnn, convolutions, tempFile, 'conv')
processLayers(cnn, full, tempFile, 'ip')
