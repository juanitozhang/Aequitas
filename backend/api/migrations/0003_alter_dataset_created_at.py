# Generated by Django 4.0.1 on 2022-02-06 21:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_rename_retraining_input_dataset_retraining_inputs'),
    ]

    operations = [
        migrations.AlterField(
            model_name='dataset',
            name='created_at',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
