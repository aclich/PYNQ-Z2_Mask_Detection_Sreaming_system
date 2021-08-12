package com.example.maskdetectiononfpga;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;

public class MainActivity2 extends AppCompatActivity {
    private WebView webView;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        Intent intent = getIntent();
        String url = intent.getStringExtra(MainActivity.EXTRA_TEXT);

        webView = (WebView) findViewById(R.id.webview);
        webView.setWebViewClient(new WebViewClient());
        WebSettings websettings = webView.getSettings();
        websettings.setJavaScriptEnabled(true);
        websettings.setUseWideViewPort(true);
        websettings.setSupportZoom(false);
        websettings.setDisplayZoomControls(false);
        websettings.setBuiltInZoomControls(true);
        webView.setInitialScale(25);
        webView.loadUrl(url);

        findViewById(R.id.button_settings).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent1 = new Intent(v.getContext(), MainActivity.class);
                startActivity(intent1);
            }
        });


    }
}