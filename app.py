from flask import Flask,request, redirect, render_template
import os
import numpy as np
import predict as predict
import cv2

app=Flask(__name__)
app.config["path"]="static"
app.config['name']=""
app.config['predict']=""
@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            image.save(os.path.join(app.config["path"],image.filename))
            img=cv2.imread(os.path.join(app.config["path"],image.filename))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            model=np.load("model10.npy",allow_pickle=True).item()
            img= predict.predict(img,model)
            app.config['predict']=image.filename.split('.')[0]+'_predict.'+image.filename.split('.')[1]
            cv2.imwrite(os.path.join('static',app.config['predict']),img)
            app.config['name']=image.filename
            return redirect(request.url)
    return render_template("upload_image.html",name=app.config['name'],src=os.path.join("../static",app.config['name']),predict_src=os.path.join('../static',app.config['predict']))




if __name__=='__main__':
    app.run(debug=True)
