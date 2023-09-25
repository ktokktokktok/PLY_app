# 必要なモジュールのインポート
import torch
from ply import transform, preprocess_image, Net 
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64
import numpy as np

def predict_images(images):
    predictions = []
    net = Net().cpu().eval()
    net.load_state_dict(torch.load(r'C:\Users\yuta1\OneDrive\デスクトップ\new_PLY\src\new3_ply.pt', map_location=torch.device('cpu')))
    for img_array in images:
        # NumPyの配列に変換
        img_array_np = np.array(img_array)
        # データ型をuint8に変換
        img_array_np = img_array_np.astype(np.uint8)
        img_pil = Image.fromarray(img_array_np)
        # 画像のチャンネル数をRGBに変更
        img_pil = img_pil.convert('RGB')
        img = transform(img_pil)
        img = img.unsqueeze(0)
        y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
        predictions.append(y)
    return predictions

# テスト画像のパスを指定して分類
#img_path = r'C:\Users\yuta1\OneDrive\デスクトップ\new_PLY\画像処理前\OK\7698_1PLYXXXXXX_DC56_L_OUT_OK_2023-0701-0053-27.bmp'
#img = Image.open(img_path)

# 画像の前処理と4分割
#processed_images = preprocess_image(img)

# ネットワークの準備
#net = Net().cpu().eval()
#net.load_state_dict(torch.load(r'C:\Users\yuta1\OneDrive\デスクトップ\new_PLY\src\new3_ply.pt', map_location=torch.device('cpu')))

# 4分割した画像を推論し、結果を集計
#predictions = predict_images(processed_images, net)

# 結果の判定
def getName(label):
    print("▲", label)
    if label == [1] * len(label):
        return "OK"
    else:
        return "NG"

# Flask のインスタンスを作成
app = Flask(__name__)

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg', 'bmp'])

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        print("■1")
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        print("■2")
        file = request.files['filename']
        # ファイルのチェック
        print("■3")
        if file and allwed_file(file.filename):

            #　画像ファイルに対する処理
            #　画像書き込み用バッファを確保
            print("★1")
            buf = io.BytesIO()
            print("★2")
            image = Image.open(file).convert('RGB')
            #　画像データをバッファに書き込む
            print("★3")
            image.save(buf, 'png')
            #　バイナリデータを base64 でエンコードして utf-8 でデコード
            print("★4")
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML 側の src  の記述に合わせるために付帯情報付与する
            print("★5")
            base64_data = 'data:image/png;base64,{}'.format(base64_str)
            # 入力された画像に対して推論
            print("★6")
            processed_images = preprocess_image(image)
            print("★7")
            predictions = predict_images(processed_images)
            print("★8", predictions)
            animalName_ = getName(predictions)
            print("★10", animalName_)
            return render_template('result.html', animalName=animalName_, image=base64_data)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')


# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)