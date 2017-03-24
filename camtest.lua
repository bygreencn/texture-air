require 'nn'
require 'image'
require 'InstanceNormalization'
require 'src/utils'
require 'torch'
require 'camera'

require 'qt'
require 'qttorch'
require 'qtwidget'

local cmd = torch.CmdLine()

cmd:option('-model_t7', '', 'Path to trained model.')
cmd:option('-cpu', false, 'use this flag to run on CPU')
cmd:option('-height', 896)
cmd:option('-width', 1600)
cmd:option('-webcam_idx', 0)
cmd:option('-webcam_fps', 60)

local params = cmd:parse(arg)

-- Load model and set type
local model = torch.load(params.model_t7)

if params.cpu then 
  tp = 'torch.FloatTensor'
else
  require 'cutorch'
  require 'cunn'
  require 'cudnn'

  tp = 'torch.CudaTensor'
  model = cudnn.convert(model, cudnn)
end

model:type(tp)
model:evaluate()
-- set camera dimension
local camera_opt = {
    idx = params.webcam_idx,
    fps = params.webcam_fps,
    height = params.height,
    width = params.width,
  }

-- init camera
local cam = image.Camera(camera_opt)
  local win = nil
  while true do
    -- Grab a frame from the webcam
    local img = cam:forward()
    -- Preprocess the frame
    local H, W = img:size(2), img:size(3)
    img = img:view(1, 3, H, W)


-- to tensor
local stylized = model:forward(img:type(tp)):double()
stylized = deprocess(stylized[1])

local img_disp = image.toDisplayTensor{
      input = stylized,
      min = 0,
      max = 1,
    }

-- scatter
if not win then
      -- On the first call use image.display to construct a window
      win = image.display(img_disp)
    else
      -- Reuse the same window
      win.image = img_out
      local size = win.window.size:totable()
      local qt_img = qt.QImage.fromTensor(img_disp)
      win.painter:image(0, 0, size.width, size.height, qt_img)
    end
end


