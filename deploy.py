from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import os
import glob



app = Flask(__name__)

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.route("/")
def index():
  #remove all old files
  files = glob.glob('static/img/*')
  visual_files = sorted(glob.glob('static/visual_img/*'))
  for f in files: 
    os.remove(f)
  for f in visual_files: 
    os.remove(f)
  return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        clss, prob, pred, convert = x_ray_pred('static/img/'+filename)
        visual_files = sorted(glob.glob('static/visual_img/*'))
        filename = 'static/img/'+filename
        return render_template('report.html', pred=pred, prob=round(prob,5), filename=filename, visual_files=visual_files, convert=convert)

        # '<h1>Predicted Class = {}</h1><h1> Prob = {}% </h1><h1>Prediction = {}</h1>'.format(clss, prob, pred)
@app.route('/example/<example>', methods=['GET', 'POST'])
def submit_example(example):
  if example =='normal':
    clss, prob, pred, convert = x_ray_pred('static/example_imgs/n_xray.jpg')
    filename = '../static/example_imgs/n_xray.jpg'
  else:
    clss, prob, pred, convert = x_ray_pred('static/example_imgs/p_xray.jpg')
    filename = '../static/example_imgs/p_xray.jpg'
  visual_files = sorted(glob.glob('static/visual_img/*'))
  return render_template('report.html', pred=pred, prob=round(prob,5), filename=filename, visual_files=visual_files, convert=convert)



#-------------------------------------------------------
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
class ResNet(nn.Module):
  def __init__(self, mid_fc_dim=100, output_dim=2):
    super(ResNet, self).__init__()
    
    self.resnet = models.resnet18(pretrained=True)
    self.resnet.layer4.requires_grad = True
    self.resnet.avgpool.requires_grad = True
    self.inp_features = self.resnet.fc.in_features
    self.resnet.fc = Identity()

    self.fc1 = nn.Linear(self.inp_features, mid_fc_dim)
    self.fc2 = nn.Linear(mid_fc_dim, output_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.3)
    self.log_softmax = nn.LogSoftmax(dim=1)
  
  def forward(self, x):
    x = self.resnet(x)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.dropout(x)
    
    x = self.fc2(x)
    x = self.log_softmax(x)
    return x

model_new = torch.load('model/trained_x_ray_classification.pth', map_location='cpu')

int_to_class = ['NORMAL', 'PNEUMONIA']
inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
   

def img_to_tensor(image_name):
  convert = False
  image = Image.open(image_name)
  if image.mode == 'L':
    image = image.convert(mode='RGB')
    convert=True

  image = inference_transform(image).float()
  image = image.unsqueeze(0) 
  return image, convert

def check_image(img):
  log_ps = model_new(img)
  ps = torch.exp(log_ps)
  top_p, top_class = ps.topk(1, dim=1)
  pred = int(top_class)
  return pred, top_p.detach().numpy().reshape(-1)[0]*100

def x_ray_pred(image):
  img, convert = img_to_tensor(image)
  visualize_cnn(img)
  pred, prob = check_image(img)
  return pred, prob, int_to_class[pred], convert

#--------------------------------------------------------
from torchvision.transforms import ToPILImage
to_img = ToPILImage()


def save_visual(output, name):
    for i in range(int(output.size(0))):
      img = to_img(output[i])
      basewidth = 150
      wpercent = (basewidth/float(img.size[0]))
      hsize = int((float(img.size[1])*float(wpercent)))
      img = img.resize((basewidth,hsize), Image.ANTIALIAS)
      img.save('static/visual_img/{}_{}.jpg'.format(name,i))

def visualize_cnn(x):
  conv1 = nn.Sequential(*list(model_new.resnet.children())[:1])(x)[0,0:10,:,:]
  layer1 = nn.Sequential(*list(model_new.resnet.children())[:5])(x)[0,:10,:,:]
  layer2 = nn.Sequential(*list(model_new.resnet.children())[:6])(x)[0,:10,:,:]

  save_visual(conv1, 'conv1')
  save_visual(layer1, 'layer1')
  save_visual(layer2, 'layer2')


# # # ------------------------------------------------------
# if __name__ == "__main__":
#   app.run(host= '0.0.0.0', debug=True)
