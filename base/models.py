# base/models.py
from django.db import models
from django.utils import timezone


class ImageModel(models.Model):
    title = models.CharField(max_length=255)
    image = models.ImageField(upload_to='images/')
    text = models.TextField(blank=True, null=True)# 이미지를 저장할 경로를 지정해주세요

    def __str__(self):
        return self.title

class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    text = models.TextField(blank=True)