from sklearn.model_selection import train_test_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from PIL import Image
import xml.etree.ElementTree as ET
import torch
import os

class CustomDataset(Dataset):
    def __init__(self, image_paths, xml_paths, class_to_idx):
        self.image_paths = image_paths
        self.xml_paths = xml_paths
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        xml_path = self.xml_paths[idx]

        # 이미지 로드
        image = Image.open(img_path).convert("RGB")

          # XML 파싱
        tree = ET.parse(xml_path)
        root = tree.getroot()

         # 객체 정보 추출
        labels = []
        boxes = []

        for obj in root.findall('object'):
            label = obj.find('name').text

        # 라벨이 class_to_idx에 없는 경우 무시
            if label not in self.class_to_idx:
               continue

            labels.append(label)

            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])

        if not boxes:
            labels.append("dummy")
            boxes.append([0, 0, 1, 1])

         # 클래스 인덱스로 변환
        labels = [self.class_to_idx[label] for label in labels]

        # PyTorch Tensor로 변환
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        return image, {"boxes": boxes, "labels": labels}




# 이미지와 XML 폴더 경로
slag_image_folder = 'C:/Users/USER/Desktop/DGU/4-2/다학제/데이터셋/01. data/Image/Slag'

slag_xml_folder = 'C:/Users/USER/Desktop/DGU/4-2/다학제/데이터셋/01. data/Label/Slag'

# 파일 목록 가져오기
slag_image_paths = [os.path.join(slag_image_folder, img) for img in os.listdir(slag_image_folder)]
slag_xml_paths = [os.path.join(slag_xml_folder, xml) for xml in os.listdir(slag_xml_folder)]

# 클래스 인덱스 매핑 생성
# 클래스 인덱스 매핑 생성 (dummy 클래스 추가)
classes = ['Slag', 'dummy']
class_to_idx = {cls: i for i, cls in enumerate(classes)}

# 데이터를 훈련 데이터와 테스트 데이터로 나누기
train_image_paths, test_image_paths, train_xml_paths, test_xml_paths = train_test_split(
    slag_image_paths, slag_xml_paths, test_size=0.2, random_state=42
)

# Collate 함수 정의 (이미지를 Tensor로 변환)
def collate_fn(batch):
    images, targets = zip(*batch)
    
    # 이미지를 Tensor로 변환
    images = [T.ToTensor()(img) for img in images]

    return images, targets

# CustomDataset 인스턴스 생성 (훈련 데이터)
train_dataset = CustomDataset(train_image_paths, train_xml_paths, class_to_idx)

# DataLoader 생성 (훈련 데이터)
train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# CustomDataset 인스턴스 생성 (테스트 데이터)
test_dataset = CustomDataset(test_image_paths, test_xml_paths, class_to_idx)

# DataLoader 생성 (테스트 데이터)
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 이미지 전처리를 위한 변환 정의
transform = T.Compose([
    T.ToTensor(),  # 이미지를 PyTorch Tensor로 변환
    # 다른 전처리 작업 추가 (크기 조정, 정규화 등)
])

# Faster R-CNN 모델 초기화
model = fasterrcnn_resnet50_fpn(pretrained=True)

# 모델을 훈련 모드로 설정
model.train()

# 옵티마이저 초기화 (Faster R-CNN에 내장된 옵티마이저 사용)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 훈련 루프
num_epochs = 10

for epoch in range(num_epochs):
    for batch in train_data_loader:
        images, targets = batch

        # 훈련 데이터를 모델에 전달하여 손실 계산
        loss_dict = model(images, targets)
        
        # 모든 손실 합산
        loss = sum(loss for loss in loss_dict.values())

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Total Loss: {loss.item()}')

# 훈련된 모델 저장
torch.save(model.state_dict(), 'C:/Users/USER/Desktop/DGU/4-2/다학제')