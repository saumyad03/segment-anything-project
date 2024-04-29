from flask import Flask, render_template, request, redirect
from transformers import SamModel, SamConfig, SamProcessor
from PIL import Image
import torch, os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
model_config = SamConfig.from_pretrained('facebook/sam-vit-base')
model = SamModel(config=model_config)
model.load_state_dict(torch.load('./intro_to_ai_model2.pth'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file:
            # filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg'))
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input.jpg')
            image = np.array(Image.open(image_path))
            inputs = processor(image, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs, multimask_output=False)
            medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
            medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
            medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
            medsam_seg_image = Image.fromarray(medsam_seg * 255)
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'output.jpg')
            medsam_seg_image.save(output_path)
            return render_template('result.html', image1_path=image_path, image2_path=output_path)

if __name__ == '__main__':
    app.run(debug=True)