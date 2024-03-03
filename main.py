from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import os
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Set upload folder

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Get the uploaded image
        image = request.files['image']
        # Secure the filename
        filename = secure_filename(image.filename)
        # Save the image to the uploads folder
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Generate a URL for the uploaded image
        image_url = url_for('static', filename='uploads/' + filename)

        img = Image.open(str('static/uploads/' + filename))

        # Grayscale conversion
        grayscale_image = img.convert("L")
        grayscale_image_path = 'static/grayscale_image.jpg'
        grayscale_image.save(grayscale_image_path)

        # Resize image
        new_height, new_width = 100, 100
        shrinked_img = img.resize((new_height, new_width))
        shrinked_image_path = 'static/shrinked_image.jpg'
        shrinked_img.save(shrinked_image_path)
        # Convert to PIL Image
        img_pil = Image.fromarray(np.array(img))
        # Horizontal Flip
        horizontal_img_path='static/horizontal_image.jpg'
        img_horizontal = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        img_horizontal.save(horizontal_img_path)

        # Vertical Flip
        vertical_img_path='static/vertical_image.jpg'
        img_vertical = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
        img_vertical.save(vertical_img_path)

        # Generate URLs for static images
        img_path = url_for('static', filename='potato_plant.jpg')
        grayscale_img_path = url_for('static', filename='grayscale_image.jpg')
        shrinked_img_path = url_for('static', filename='shrinked_image.jpg')
        vertical_img_path = url_for('static', filename='vertical_image.jpg')
        horizontal_img_path = url_for('static', filename='horizontal_image.jpg')

        return render_template('index.html', image_url=image_url, img_path=img_path, grayscale_img_path=grayscale_img_path,shrinked_img_path=shrinked_img_path,horizontal_img_path=horizontal_img_path, vertical_img_path=vertical_img_path )
    return render_template('index.html')
@app.route('/')
def show_images():
    # Load original image
    img = Image.open('static/potato_plant.jpg')

    # Grayscale conversion
    grayscale_image = img.convert("L")
    grayscale_image_path = 'static/grayscale_image.jpg'
    grayscale_image.save(grayscale_image_path)

    # Resize image
    new_height, new_width = 100, 100
    shrinked_img = img.resize((new_height, new_width))
    shrinked_image_path = 'static/shrinked_image.jpg'
    shrinked_img.save(shrinked_image_path)
    img_pil = Image.fromarray(np.array(img))
    # Horizontal Flip
    horizontal_img_path = 'static/horizontal_image.jpg'
    img_horizontal = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    img_horizontal.save(horizontal_img_path)

    # Vertical Flip
    vertical_img_path = 'static/vertical_image.jpg'
    img_vertical = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
    img_vertical.save(vertical_img_path)
    # Generate URLs for static images
    img_path = url_for('static', filename='potato_plant.jpg')
    grayscale_img_path = url_for('static', filename='grayscale_image.jpg')
    shrinked_img_path = url_for('static', filename='shrinked_image.jpg')
    vertical_img_path=url_for('static', filename='vertical_image.jpg')
    horizontal_img_path=url_for('static', filename='horizontal_image.jpg')

    return render_template('index.html', img_path=img_path,shrinked_img_path=shrinked_img_path, grayscale_img_path=grayscale_img_path, horizontal_img_path=horizontal_img_path, vertical_img_path=vertical_img_path )

if __name__ == '__main__':
    app.run(debug=True)
