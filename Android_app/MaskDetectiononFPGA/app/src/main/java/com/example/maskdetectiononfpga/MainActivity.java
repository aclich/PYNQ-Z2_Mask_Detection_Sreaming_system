package com.example.maskdetectiononfpga;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;

public class MainActivity extends AppCompatActivity {
    public static final String EXTRA_TEXT = "com.example.maskdetectiononfpga.EXTRA_TEXT";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button_save = (Button) findViewById(R.id.button_save);
        button_save.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                openActivity2();
            }
        });

    }
    public void openActivity2(){
        EditText edittext_url = (EditText) findViewById(R.id.edittext_url);
        String url = edittext_url.getText().toString();


        Intent intent = new Intent(this, MainActivity2.class);
        intent.putExtra(EXTRA_TEXT, url);
        startActivity(intent);
    }
}