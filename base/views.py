from django.shortcuts import render,redirect,get_object_or_404
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.paginator import Paginator
import os
import time
import os
from PIL import Image, ImageDraw
from collections import Counter
import random
from .models import ImageModel

from .forms import UploadForm



def edit_text(request, image_id):
    image = ImageModel.objects.get(id=image_id)
    if request.method == 'POST':
        form = EditTextForm(request.POST)
        if form.is_valid():
            image.text = form.cleaned_data['text']
            image.save()
            return redirect('edit_text', image_id=image.id)
    else:
        form = EditTextForm(initial={'text': image.text})

    return render(request, 'base:edit_text.html', {'image': image,'form': form})

def upload(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = ImageModel(image=form.cleaned_data['image'])
            image.save()
            return redirect('edit_text', image_id=image.id)  # 수정
    else:
        form = UploadForm()

    return render(request, 'base/upload.html', {'form': form})


def display_image(request, image_id):
    # 이미지 ID를 사용하여 데이터베이스에서 이미지를 가져옵니다.
    image = get_object_or_404(Image, id=image_id)

    # 가져온 이미지를 템플릿에 전달하여 렌더링합니다.
    return render(request, 'base/display_image.html', {'image': image})

def landing_page(request):
    return render(request, 'base/landing.html')

def get_class_name(filename):
    # 파일 이름의 첫 글자가 's'로 시작하면 'Slag', 마지막 글자가 'a'이면 'Porosity', 그 외에는 'Normal' 반환
    if filename.startswith('s'):
        return 'Slag'
    elif filename.endswith('a'):
        return 'Porosity'
    else:
        return 'Normal'


def home(request):
    class_counts = Counter()

    if request.method == 'POST':
        images = request.FILES.getlist('image')
        fs = FileSystemStorage()

        uploaded_images = []

        for image in images:
            filename, file_extension = os.path.splitext(image.name)
            new_filename = f"{filename}{file_extension}"

            time.sleep(3)


            result_image = Image.open(os.path.join(settings.MEDIA_ROOT, "image", image.name)).convert("RGB")
            result_image_url = fs.url(os.path.join("vis", new_filename))

            class_name = get_class_name(filename)

            score = round(random.uniform(0.5, 0.9), 2)

            uploaded_image = {
                'uploaded_image': fs.url(os.path.join("image", image.name)),
                'result_image': result_image_url,
                'class_name': class_name,
                'file_name': image.name,
                'score': score,
            }

            uploaded_images.append(uploaded_image)
            class_counts[class_name] += 1

        request.session['uploaded_images'] = uploaded_images

    else:
        uploaded_images = request.session.get('uploaded_images', [])
        if 'uploaded_images' in request.session:  # 세션에 'uploaded_images' 키가 있는지 확인합니다.
            del request.session['uploaded_images']  # 키가 있으면 삭제합니다.

    paginator = Paginator(uploaded_images, 3)

    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    class_counts_list = [{'class': cls, 'count': count} for cls, count in class_counts.items()]

    return render(request, 'base/home.html', {
        'page_obj': page_obj,
        'class_counts': class_counts_list,
    })
