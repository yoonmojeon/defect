# Generated by Django 4.2.7 on 2023-12-20 17:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0003_rename_uploaded_image_image_file_image_uploaded_at'),
    ]

    operations = [
        migrations.RenameField(
            model_name='image',
            old_name='file',
            new_name='image',
        ),
        migrations.RemoveField(
            model_name='image',
            name='uploaded_at',
        ),
        migrations.AddField(
            model_name='imagemodel',
            name='text',
            field=models.TextField(blank=True, null=True),
        ),
    ]
