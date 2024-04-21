# base/urls.py

from django.urls import path,include
from .views import home
from django.conf import settings
from django.contrib.staticfiles.views import serve
from django.urls import re_path
from . import views

app_name = 'base'



urlpatterns = [
    path('', views.landing_page, name='landing'),
    path('home/', views.home, name='home'),
    path('display_image/<int:image_id>/', views.display_image, name='display_image'),
    path('upload/', views.upload, name='upload'),
    path('edit_text/<int:image_id>/', views.edit_text, name='edit_text'),
]

if settings.DEBUG is False:
    urlpatterns += [
        re_path(r'^static/(?P<path>.*)$', serve),
    ]