from flask import Flask, request, redirect, url_for, render_template
import frs, pickle, os
from keras.models import load_model

app = Flask(__name__)

# Specify the directory to save uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/products')
def products():
    return render_template('products.html')

@app.route('/fashion')
def fashion():
    return render_template('fashion.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')


# Shirts pages
@app.route('/red')
def red():
    # return ("red shirts page")
    return render_template('red.html')

@app.route('/green')
def green():
    # return ("green shirts page")
    return render_template('green.html')

@app.route('/kurtas')
def kurtas():
    # return (" kurtas page")
    return render_template('kurtas.html')

@app.route('/sarees')
def sarees():
    # return (" sarees page")
    return render_template('sarees.html')


@app.route('/yellow')
def yellow():
    # return (" sarees page")
    return render_template('yellow.html')

@app.route('/black')
def black():
    # return (" sarees page")
    return render_template('black.html')


@app.route('/Purple')
def Purple():
    # return (" sarees page")
    return render_template('Purple.html')


@app.route('/blue')
def blue():
    # return (" sarees page")
    return render_template('blue.html')


@app.route('/casual')
def  casual():
    # return (" sarees page")
    return render_template('casual.html')


@app.route('/cameras')
def cameras():
    # return (" sarees page")
    return render_template('cameras.html')

# @app.route('/uploadImage', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return redirect(request.url)
#     file = request.files['file']
#     if file.filename == '':
#         return redirect(request.url)
#     if file:
#         filename = file.filename
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         return 'File uploaded successfully'

@app.route('/uploadImage',methods=["GET","POST"])
def uploadImage():
    file=request.files['file']
    file_path = 'static/uploads/' + file.filename
    file.save(file_path)

    # from  pkl files (features.pkl, names.pkl)
    with open('names.pkl','rb') as f: all_image_names = pickle.load(f)
    with open('features.pkl','rb') as f: all_features = pickle.load(f)
    # save model
    savedModel=load_model(r'D:\finalweights.h5')

    input_image_path = file_path
    # Adjust to an actual file path
    k = frs.recommend_fashion(input_image_path, all_features, all_image_names, savedModel, top_n=4)

    t=[]
    for i in k:
        t.append(i.replace("D:\\images\\", ''))
    print(t)

    # t=[]
    # for i in k:
    #     file_path = 'static/recommendations/' + i[13:]
    #     t.append(i[13:])
    #     file.save(file_path)

    return render_template('upload.html',res=t)

if __name__ == '__main__':
    app.run(port=5000, debug=True)