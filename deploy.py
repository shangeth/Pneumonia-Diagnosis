from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import os
import glob



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route("/")
def index():
  #remove all old files
  files = glob.glob('static/img/*')
  visual_files = sorted(glob.glob('static/visual_img/*'))
  for f in files: 
    os.remove(f)
  for f in visual_files: 
    os.remove(f)

  return render_template('index2.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        clss, prob, pred, convert = x_ray_pred('static/img/'+filename)
        filename = 'static/img/'+filename
        return render_template('report.html', pred=pred, prob=round(prob,5), filename=filename, convert=convert)

        # '<h1>Predicted Class = {}</h1><h1> Prob = {}% </h1><h1>Prediction = {}</h1>'.format(clss, prob, pred)


@app.route('/example/<example>', methods=['GET', 'POST'])
def submit_example(example):
  example_dict = {'n1':'n1.jpg', 'n2':'n2.jpeg', 'n3':'n3.jpg', 'p1':'p1.jpg', 'p2':'p2.jpg', 'p3':'p3.jpeg'}
  clss, prob, pred, convert = x_ray_pred('static/example_imgs/'+str(example_dict[example]))
  filename = '../static/example_imgs/'+str(example_dict[example])
  return render_template('report.html', pred=pred, prob=round(prob,5), filename=filename, convert=convert)



#-------------------------------------------------------
from PIL import Image
import torch
import torchvision
from torchvision import transforms, models
import torch.nn as nn
from resnet_models import *


model =  ResNet()
model.load_state_dict(torch.load('model/best_model.pt', map_location='cpu'))    
model.eval()




def img_to_tensor(image_name):
  inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  convert = False
  image = Image.open(image_name)
  if image.mode == 'L':
    image = image.convert(mode='RGB')
    convert=True

  image = inference_transform(image).float()
  image = image.unsqueeze(0) 
  return image, convert

def check_image(img):
  log_ps = model(img)
  ps = torch.exp(log_ps)
  top_p, top_class = ps.topk(1, dim=1)
  pred = int(top_class)
  return pred, top_p.detach().numpy().reshape(-1)[0]*100

def x_ray_pred(image):
  
  int_to_class = ['NORMAL', 'PNEUMONIA']
  img, convert = img_to_tensor(image)
  pred, prob = check_image(img)

  
  return pred, prob, int_to_class[pred], convert

#--------------------------------------------------------






# # # ------------------------------------------------------
if __name__ == "__main__":
  app.run(host= '0.0.0.0', debug=True)
