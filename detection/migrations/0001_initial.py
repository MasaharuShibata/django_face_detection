# Generated by Django 3.1.2 on 2021-04-30 08:13

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Document',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('description', models.CharField(blank=True, max_length=255)),
                ('photo', models.ImageField(default='SOME STRING', upload_to='gallery/')),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('gray', models.ImageField(default='Not Set', upload_to='')),
            ],
        ),
    ]
