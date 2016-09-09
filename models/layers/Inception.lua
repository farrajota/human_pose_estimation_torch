--[[
    Inception modules from inception v4. 
    Source: https://arxiv.org/pdf/1602.07261v2.pdf
]]

local conv = nn.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = nn.ReLU

-- round to the nearest multiplier of 16
local function nearest16(input)
  if input >=16 then
      local rest = input%16
      if rest >= 8 then
          return input + rest
      else
          return input - rest
      end
  else
      return 16
  end
end

-- conv + bachnorm + relu
local function ConvBN(numIn,numOut, kernelx, kernely, stridex, stridey, padx, pady)
  return nn.Sequential()
      :add(conv(numIn,numOut, kernelx, kernely,  stridex, stridey, padx, pady))
      :add(batchnorm(numOut))
      :add(relu(true))
end

-- Inception-Resnet-v2-A (figure 16)
local function convBlockA(numIn,numOut)
  local numReduct = math.ceil(numIn/3) - math.ceil(numIn/3)%16
  local branch1 = ConvBN(numIn,numReduct,1,1,1,1)
  local branch2 = nn.Sequential()
      :add(ConvBN(numIn,numReduct,1,1,1,1))
      :add(ConvBN(numReduct,numReduct,3,3,1,1,1,1))
  local branch3 = nn.Sequential()
      :add(ConvBN(numIn,numReduct,1,1,1,1))
      :add(ConvBN(numReduct,numReduct*1.5,3,3,1,1,1,1))
      :add(ConvBN(numReduct*1.5,numReduct*2,3,3,1,1,1,1))
  local convLin = ConvBN(numReduct*4,numOut,1,1,1,1)
  
  return nn.Sequential()
      :add(nn.ConcatTable()
          :add(branch1)
          :add(branch2)
          :add(branch3))
      :add(nn.JoinTable(2))
      :add(convLin)
end

-- Inception-Resnet-v2-B (figure 17)
local function convBlockB(numIn,numOut)
  local numReduct = nearest16(math.ceil(numIn/3)) -- keep the minimum size at least 16
  local numReduct2 = nearest16((numReduct+numIn/2)/2)
  local branch1 = ConvBN(numIn,numIn/2,1,1,1,1)
  local branch2 = nn.Sequential()
      :add(ConvBN(numIn,numReduct,1,1,1,1))
      :add(ConvBN(numReduct,numReduct2,7,1,1,1,3,0))
      :add(ConvBN(numReduct2,numIn/2,1,7,1,1,0,3))
  local convLin = ConvBN(numIn,numOut,1,1,1,1)
  
  return nn.Sequential()
      :add(nn.ConcatTable()
          :add(branch1)
          :add(branch2))
      :add(nn.Concat())
      :add(convLin)
end

-- Inception-Resnet-v2-C (figure 19)
local function convBlockC(numIn,numOut)
  local nfeats1 = nearest16(math.ceil((numIn/2)*(4/3))) -- keep the minimum size at least 16
  local nfeats2 = (numIn/2+nfeats1)/2
  local branch1 = ConvBN(numIn,numIn/2,1,1,1,1)
  local branch2 = nn.Sequential()
      :add(ConvBN(numIn,numIn/2,1,1,1,1))
      :add(ConvBN(numIn/2,nfeats2,3,1,1,1,1,0))
      :add(ConvBN(nfeats2,nfeats1,1,3,1,1,0,1))
  local convLin = ConvBN(nfeats1+numIn/2,numOut,1,1,1,1)
  
  return nn.Sequential()
      :add(nn.ConcatTable()
          :add(branch1)
          :add(branch2))
      :add(nn.Concat())
      :add(convLin)
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(conv(numIn,numOut,1,1))
    end
end

-- Residual block
function InceptionResidual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlockA(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end

