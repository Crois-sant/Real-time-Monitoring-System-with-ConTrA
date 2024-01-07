import json
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
import redis
from pydantic import BaseModel
import torch
import pickle
import numpy as np
from conformer_MSCRED import ConformerMSCRED, MinMaxScaler, calculate_signature_matrix_dataset

app = FastAPI()

def get_result(model, input_data, max_, min_, columns):
    values = []
    for input_dict in input_data:
        value = [input_dict[key] for key in input_dict if key in columns]
        value = list(map(float, value))
        values.append(value)
    values = np.array(values)

    scaled_values, _, _  = MinMaxScaler(values, max_=max_, min_=min_)

    X, y = calculate_signature_matrix_dataset(scaled_values, lags=[10, 30, 60], stride=1, num_timesteps=5)
    X = X.transpose(0, 4, 1, 2, 3)
    y = y.transpose(0, 3, 1, 2)
    X = torch.Tensor(X)

    with torch.no_grad():
        pred = model(X).detach().cpu().numpy()

    residual_matrix = y - pred
    square_sum_rm = np.sum(residual_matrix**2, axis=1)
    diagonal_rm = square_sum_rm[0].diagonal()
    top3_indices = np.argsort(diagonal_rm)[:-4:-1]
    root_causes = [columns[idx] for idx in top3_indices]
    
    square_sum_rm = np.sum(square_sum_rm, axis=1)
    mse = np.sum(square_sum_rm, axis=1) / (pred.shape[1] * pred.shape[2] * pred.shape[3])
    threshold =  7.22644e-05
    if mse[0] > threshold:
        score = 0
    else:
        score = 100 - ((mse[0]/threshold)*100)

    return root_causes, score

# WebSocket 메시지 모델
class Message(BaseModel):
    data: dict

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = './best_weight.pt'
model = ConformerMSCRED(device=device).to(device)

if device == 'cpu':
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
else:
    model.load_state_dict(torch.load(ckpt_path))

with open('./input_pipeline/scale_factor.pkl', 'rb') as f:
    max_, min_ = pickle.load(f)

with open('./input_pipeline/columns.pkl', 'rb') as f:
    columns = pickle.load(f)

# WebSocket 연결을 위한 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            input_data = json.loads(data)
            try:
                input_list = input_data['message']
                root_causes, score = get_result(model, input_list, max_, min_, columns)
                print(f'root_causes: {root_causes}, score: {float(score)}')
                # 결과 전송
                await websocket.send_text(json.dumps({'prediction': {"root_causes": root_causes, "safety_score": score}}))
            except Exception as e:
                await websocket.send_text(json.dumps({'error': f'예측 오류: {str(e)}'}))
    except WebSocketDisconnect:
        pass
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
