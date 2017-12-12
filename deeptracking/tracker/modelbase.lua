--
-- User: mathieu
-- Date: 19/12/16
-- Time: 8:34 AM
--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'rnn'
require 'cutorch'
require 'cunn'
require 'math'

local ModelBase = torch.class('ModelBase')

function ModelBase:__init(backend, optimfunc, device)
    self.net = nil
    self.backend = backend
    if device == nil then self.device = 1 else self.device = device end
    if self.backend == "cuda" then
        print(string.format("Using Device %d", self.device))
        cutorch.setDevice(self.device)
    end
    if optimfunc == "sgd" then
       self.optimFunction = optim.sgd
    elseif optimfunc == "adadelta" then
        self.optimFunction = optim.adadelta
    else
        self.optimFunction = optim.adam
    end
    self.config = {
        learningRate = 0.005,
        learningRateDecay = 0,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-08,
        momentum = 0.9,
        dampening = 0,
        nesterov = 0.9,
    }
end

function ModelBase:set_configs(dict)
    for k, v in pairs(dict) do
        self.config[k] = v
    end
end

function ModelBase:get_configs(name)
    return self.config[name]
end

function ModelBase:model_string()
    local backend_string = string.format("Backend : %s", self.backend)
    local net_string = string.format("%s\n%s", backend_string, self.net)
    return net_string
end

function ModelBase:memory_info_string()
    require 'cutorch'
    local idx = cutorch.getDevice()
    local freeMemory, totalMemory = cutorch.getMemoryUsage(idx)
    return string.format("Free memory : %f Gb, Total memory : %f Gb", freeMemory/1073741824.0, totalMemory/1073741824.0)
end

-- Convert tensor based on backend requested
function ModelBase:setup_tensor(ref, buffer)
    local localOutput = buffer
    if self.backend == 'cpu' then
        localOutput = ref
    else
        localOutput = localOutput or ref:clone()
        if torch.type(localOutput) ~= 'torch.CudaTensor' then
            localOutput = localOutput:cuda()
        end
        localOutput:resize(ref:size())
        localOutput:copy(ref)
    end
    return localOutput
end

function ModelBase:set_backend(module)
    if self.backend == 'cuda' then
        module = module:cuda()
    else
        module = module:float()
    end
    return module
end

function ModelBase:convert_backend(backend)
    self.backend = backend
    if self.backend == 'cuda' then
        self.net = self.net:cuda()
        for k,v in pairs(self.config) do
            if type(v) == "userdata" then
                self.config[k] = v:cuda()
            end
        end
    else
        self.net = self.net:float()
            for k,v in pairs(self.config) do
            if type(v) == "userdata" then
                self.config[k] = v:float()
            end
        end
    end
    return
end

function ModelBase:convert_inputs(inputs)
    -- this function is used when you have particular inputs, it handles backend transfer and any formating to the input data
    error("convert_inputs not defined!")
end

function ModelBase:convert_outputs(outputs)
    -- convert forward outputs so it can be handled in python
    error("convert_outputs not defined!")
end

function ModelBase:compute_criterion(forward_input, label)
    -- compute the criterion given the output of forward and labels, returns a dict with losses :
    -- label : the generic loss used for trainning algorithm
    -- user_defined_loss : any other loss.
    error("compute_criterion not defined!")
end

function ModelBase:extract_features()
    -- This function return a dict containning layers activations. By default it will return nil
    return nil
end

function ModelBase:extract_grad_statistic()
    -- This function return a dict containning gradient information after backward pass.
    return nil
end

function ModelBase:on_train()
    -- Will be called when train is called. can be reimplemented by subclasses
end

function ModelBase:init_model()
    self.net = self:set_backend(self.net)
    self.params, self.gradParams = self.net:getParameters()
end

function ModelBase:train(inputs, labels)
    self.net:training()
    self:on_train()
    local func = function(x)
        self.gradParams:zero()
        local converted_inputs = self:convert_inputs(inputs)
        local output = self.net:forward(converted_inputs)
        losses, f_grad = self:compute_criterion(output, labels)
        self.net:backward(converted_inputs, f_grad)
       return losses['label'], self.gradParams
    end
    self.optimFunction(func, self.params, self.config)
    return losses
end

function ModelBase:test(inputs)
    collectgarbage(); collectgarbage()
    self.net:evaluate()
    local converted_inputs = self:convert_inputs(inputs)
    local output = self.net:forward(converted_inputs)
    -- Pytorch does not support gpu tensor output
    return self:convert_outputs(output, "cpu")
end

function ModelBase:loss_function(prediction, truth)
    local prediction_b = self:convert_outputs(prediction, self.backend)
    self.truthTensor = self:setup_tensor(truth, self.truthTensor)
    losses, f_grad = self:compute_criterion(prediction_b, self.truthTensor)
    return losses
end

function ModelBase:save(path, suffix)
    suffix = suffix == nil and "" or suffix
    self.net:clearState()
    local model = self.net:clone():float()
    torch.save(path..suffix..".t7", model)
    torch.save(path..suffix.."_optim.t7", self.config)
end

function ModelBase:load(path)
    self.net = torch.load(path..".t7")
    self.config = torch.load(path.."_optim.t7")
    self:init_model()
end

