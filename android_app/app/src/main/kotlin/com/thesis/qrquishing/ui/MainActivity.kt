package com.thesis.qrquishing.ui

import android.Manifest
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.ViewModelProvider
import com.journeyapps.barcodescanner.ScanContract
import com.journeyapps.barcodescanner.ScanOptions
import com.thesis.qrquishing.ai.ModelProvider
import com.thesis.qrquishing.databinding.ActivityMainBinding
import com.thesis.qrquishing.permission.PermissionValidator.hasCameraPermission
import com.thesis.qrquishing.viewmodel.MainViewModel
import com.thesis.qrquishing.viewmodel.MainViewModelFactory

/**
 * Main activity: scans a QR code, runs TFLite inference, and warns the user
 * if the decoded URL is classified as malicious or uncertain.
 */
class MainActivity : AppCompatActivity() {
    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startScan() else uiHelper.showPermissionDeniedDialog()
        }

    private lateinit var viewModel: MainViewModel
    private lateinit var uiHelper: UiHelper
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setupBinding()
        setupViewModel()
        setupUi()
        setupObservers()
        setupListeners()
    }

    private fun setupListeners() {
        binding.btnScan.setOnClickListener { startScanWithPermissionCheck() }
        binding.executeLocal.setOnClickListener { viewModel.onQrScanned(binding.LocalRunText.text.toString()) }
    }

    private fun setupObservers() {
        viewModel.result.observe(this) { result ->
            uiHelper.updateResult(result.url, result.verdict, result.confidence)
        }

        viewModel.warningEvent.observe(this) { event ->
            event.getContentIfNotHandled()?.let { result ->
                uiHelper.showWarningDialog(result.url, result.verdict, result.confidence)
            }
        }
    }

    private fun setupViewModel() {
        val factory = MainViewModelFactory(ModelProvider.create(this))
        viewModel = ViewModelProvider(this, factory)[MainViewModel::class.java]
    }

    private fun setupUi() {
        uiHelper = UiHelper(this, binding)
        uiHelper.init()
    }

    private fun setupBinding() {
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
    }

    private fun startScanWithPermissionCheck() {
        if (hasCameraPermission(this)) startScan()
        else requestCameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun startScan() {
        val options = ScanOptions().apply {
            setDesiredBarcodeFormats(ScanOptions.QR_CODE)
            setPrompt("Scan a QR code")
        }
        scanLauncher.launch(options)
    }
    private val scanLauncher = registerForActivityResult(ScanContract()) { result ->
        result.contents?.takeIf(String::isNotBlank)?.let(viewModel::onQrScanned)
    }
}
