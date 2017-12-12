--
-- User: mathieu
-- Date: 19/12/16
-- Time: 2:56 PM
-- Description : All channels concatenation
--

require 'deeptracking.tracker.modelbase'

local RGBDTracker = torch.class('RGBDTracker', 'ModelBase')

function RGBDTracker:__init(backend, optimfunc)
    ModelBase.__init(self, backend, optimfunc)

end

function RGBDTracker:build_convo(input_channels, c1_filters, c2_filters, final_size)
    local c1_filter_size = 5
    local c2_filter_size = 3
    local max_pooling_size = 2

    local first = nn:Sequential()
    first:add(nn.SpatialConvolution(4, c1_filters, c1_filter_size, c1_filter_size))
    first:add(nn.SpatialBatchNormalization(c1_filters))
    first:add(nn.ELU())
    first:add(nn.SpatialMaxPooling(max_pooling_size, max_pooling_size, max_pooling_size, max_pooling_size))

    local input = nn:ParallelTable()
    input:add(first)
    input:add(first:clone())

    local convo = nn:Sequential()
    convo:add(input)
    convo:add(nn.JoinTable(2))

    convo:add(nn.SpatialConvolution(c1_filters * 2, c2_filters, c2_filter_size, c2_filter_size))
    convo:add(nn.SpatialBatchNormalization(c2_filters))
    convo:add(nn.ELU())
    --convo:add(nn.SpatialDropout(0.2))
    convo:add(nn.SpatialMaxPooling(max_pooling_size, max_pooling_size, max_pooling_size, max_pooling_size))

    convo:add(nn.SpatialConvolution(c2_filters, c2_filters, c2_filter_size, c2_filter_size))
    convo:add(nn.SpatialBatchNormalization(c2_filters))
    convo:add(nn.ELU())
    --convo:add(nn.SpatialDropout(0.2))
    convo:add(nn.SpatialMaxPooling(max_pooling_size, max_pooling_size, max_pooling_size, max_pooling_size))

    convo:add(nn.SpatialConvolution(c2_filters, c2_filters, c2_filter_size, c2_filter_size))
    convo:add(nn.SpatialBatchNormalization(c2_filters))
    convo:add(nn.ELU())
    --convo:add(nn.SpatialDropout(0.2))
    convo:add(nn.SpatialMaxPooling(max_pooling_size, max_pooling_size, max_pooling_size, max_pooling_size))
    convo:add(nn.View(-1, c2_filters * final_size))

    return convo
end

function RGBDTracker:build_model()
    local linear_size = self.config["linear_size"]
    local c1_filter_qty = self.config["convo1_size"]
    local c2_filter_qty = self.config["convo2_size"]
    local input_size = self.config["input_size"]
    local view = math.floor((((((((input_size-4)/2)-2)/2)-2)/2)-2)/2) -- todo should not be hardcoded..
    local view_size = view * view

    local convo = self:build_convo(8, c1_filter_qty, c2_filter_qty, view_size)

    local merge = nn:ParallelTable()
    merge:add(convo)
    merge:add(nn.Identity())

    self.net = nn:Sequential()
    self.net:add(merge)
    self.net:add(nn.JoinTable(1, 1))
    self.net:add(nn.Dropout(0.5))
    self.net:add(nn.Linear(c2_filter_qty * view_size + 4, linear_size))
    self.net:add(nn.ELU())
    self.net:add(nn.Linear(linear_size, 6))
    self.net:add(nn.Tanh())
end

function RGBDTracker:convert_inputs(inputs)
    self.inputTensor = self:setup_tensor(inputs[1], self.inputTensor)
    self.priorTensor = self:setup_tensor(inputs[2][{ {},{4,7} }], self.priorTensor)
    local ret = {{self.inputTensor[{{}, {1,4}, {}, {}}], self.inputTensor[{{}, {5,8}, {}, {}}] }, self.priorTensor }
    --local ret = {self.inputTensor[{{}, {1,4}, {}, {}}], self.inputTensor[{{}, {5,8}, {}, {}}] }

    return ret
end

function RGBDTracker:convert_outputs(outputs, backend)
    local ret
    if backend == "cuda" then
        ret = outputs:cuda()
    else
        ret = outputs:float()
    end
    return ret
end

function RGBDTracker:compute_criterion(forward_input, label)
    self.labelTensor = self:setup_tensor(label, self.labelTensor)
    if self.crit == nil then
        self.crit = self:set_backend(nn.MSECriterion())
    end
    local label_loss = self.crit:forward(forward_input, self.labelTensor)
    self.label_grad = self.crit:backward(forward_input, self.labelTensor)
    return {label=label_loss}, self.label_grad
end

function RGBDTracker:get_feature()
    return self.net:findModules('nn.Linear')[1].output:float()
end

function RGBDTracker:extract_grad_statistic()
    local rotation_view = self.label_grad[{{},{4, 6}}]:float():abs()
    local grad_rot_mean = torch.mean(rotation_view)
    y, i = torch.median(rotation_view, 1)
    local grad_rot_median = torch.mean(y)
    local grad_rot_min = torch.min(rotation_view)
    local grad_rot_max = torch.max(rotation_view)

    local translation_view = self.label_grad[{{},{1, 3}}]:float():abs()
    local grad_trans_mean = torch.mean(translation_view)
    y, i = torch.median(translation_view, 1)
    local grad_trans_median = torch.mean(y)
    local grad_trans_min = torch.min(translation_view)
    local grad_trans_max = torch.max(translation_view)
    return {{grad_rot_mean=grad_rot_mean, grad_rot_median=grad_rot_median, grad_rot_min=grad_rot_min, grad_rot_max=grad_rot_max},
            {grad_trans_mean=grad_trans_mean, grad_trans_median=grad_trans_median, grad_trans_min=grad_trans_min, grad_trans_max=grad_trans_max}}
end


