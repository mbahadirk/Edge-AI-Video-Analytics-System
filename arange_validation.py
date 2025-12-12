import os
import shutil
import random
import yaml

# Veri setinin ana yolu (Hata aldığın klasör yolu)
dataset_path = "/content/drive/MyDrive/AI_Assessment_Project/Hard-Hat-Workers-2"

# Klasör yolları
train_images_path = os.path.join(dataset_path, "train", "images")
train_labels_path = os.path.join(dataset_path, "train", "labels")
valid_images_path = os.path.join(dataset_path, "valid", "images")
valid_labels_path = os.path.join(dataset_path, "valid", "labels")

# Klasörleri oluştur
os.makedirs(valid_images_path, exist_ok=True)
os.makedirs(valid_labels_path, exist_ok=True)

# Resim listesini al
images = [f for f in os.listdir(train_images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
total_images = len(images)
val_count = int(total_images * 0.20)  # %20 doğrulama için ayır

print(f"📂 Toplam Resim: {total_images}")
print(f"🔄 Taşınacak Resim Sayısı (Valid): {val_count}")

# Rastgele seç
random.seed(42) # Her seferinde aynı ayrımı yapsın
val_images = random.sample(images, val_count)

# Taşıma işlemi
move_counter = 0
for img_name in val_images:
    # Dosya isimleri
    label_name = os.path.splitext(img_name)[0] + ".txt"
    
    src_img = os.path.join(train_images_path, img_name)
    dst_img = os.path.join(valid_images_path, img_name)
    
    src_lbl = os.path.join(train_labels_path, label_name)
    dst_lbl = os.path.join(valid_labels_path, label_name)
    
    # Resmi taşı
    shutil.move(src_img, dst_img)
    
    # Etiketi taşı (Eğer varsa)
    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)
    
    move_counter += 1

print(f"✅ {move_counter} adet resim ve etiket 'valid' klasörüne taşındı.")

# 2. data.yaml dosyasını güncelleme
yaml_path = os.path.join(dataset_path, "data.yaml")

# Mevcut yaml'ı oku veya yenisini oluştur
data_config = {
    'path': dataset_path,
    'train': 'train/images',
    'val': 'valid/images',
    # Sınıf isimlerini mevcut yaml'dan almaya çalışalım, yoksa default yazarız
    'names': {0: 'head', 1: 'helmet', 2: 'person'} 
}

# Eğer eski yaml varsa oradaki names bilgisini koruyalım
if os.path.exists(yaml_path):
    with open(yaml_path, 'r') as f:
        old_yaml = yaml.safe_load(f)
        if 'names' in old_yaml:
            data_config['names'] = old_yaml['names']
        if 'nc' in old_yaml:
            data_config['nc'] = old_yaml['nc']

# Yeni yaml'ı kaydet
with open(yaml_path, 'w') as f:
    yaml.dump(data_config, f, default_flow_style=False)

print(f"📝 data.yaml güncellendi: {yaml_path}")