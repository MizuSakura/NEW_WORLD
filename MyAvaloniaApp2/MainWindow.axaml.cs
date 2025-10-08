using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Threading;
using System;
using System.Net.Http;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.Globalization;

namespace MyAvaloniaApp2;

// "แบบฟอร์ม" สำหรับ Single Simulation
public class SimulationParams
{
    public float R { get; set; }
    public float C { get; set; }
    public float dt { get; set; }
    public string control_mode { get; set; }
    public float setpoint_level { get; set; }
    public float time_sim { get; set; }
    public string signal_type { get; set; }
    public float amplitude { get; set; }
    public float duty { get; set; }
    public float freq { get; set; }
}

// "แบบฟอร์ม" สำหรับ Batch Simulation
public class BatchSimulationParams
{
    public float R { get; set; }
    public float C { get; set; }
    public float time_sim { get; set; }
    public float amplitude { get; set; }
    public float duty_start { get; set; }
    public float duty_end { get; set; }
    public int duty_steps { get; set; }
    public float freq_start { get; set; }
    public float freq_end { get; set; }
    public int freq_steps { get; set; }
}


public partial class MainWindow : Window
{
    private static readonly HttpClient _httpClient = new();
    private ClientWebSocket? _webSocket;
    private CancellationTokenSource? _cancellationTokenSource;

    public MainWindow()
    {
        InitializeComponent();
        _httpClient.BaseAddress = new Uri("http://127.0.0.1:8000");
        AppendLog("System Ready. Waiting for command.");
    }
     private T ParseValue<T>(string controlName) where T : struct
    {
        var textBox = this.FindControl<TextBox>(controlName);
        if (textBox == null || string.IsNullOrWhiteSpace(textBox.Text))
        {
            throw new ArgumentNullException(controlName, "TextBox is missing or empty.");
        }
        // ใช้ CultureInfo.InvariantCulture เพื่อให้แน่ใจว่าใช้ . เป็นทศนิยมเสมอ
        return (T)Convert.ChangeType(textBox.Text, typeof(T), CultureInfo.InvariantCulture);
    }

    private string GetComboBoxValue(string controlName)
    {
        var comboBox = this.FindControl<ComboBox>(controlName);
        if (comboBox?.SelectedItem is ComboBoxItem selectedItem)
        {
            return selectedItem.Content.ToString();
        }
        throw new InvalidOperationException($"ComboBox '{controlName}' has no selection.");
    }


    // --- Event Handler สำหรับปุ่ม "อาหารตามสั่ง" ---
     private async void StartSingleSimulationButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            var simParams = new SimulationParams
            {
                R = ParseValue<float>("RTextBox"),
                C = ParseValue<float>("CTextBox"),
                dt = ParseValue<float>("DtTextBox"),
                setpoint_level = ParseValue<float>("SetpointTextBox"),
                control_mode = GetComboBoxValue("ControlModeComboBox"),
                time_sim = ParseValue<float>("TimeSimTextBox"),
                signal_type = GetComboBoxValue("SignalTypeComboBox"),
                amplitude = ParseValue<float>("AmplitudeTextBox"),
                duty = ParseValue<float>("DutyTextBox"),
                freq = ParseValue<float>("FreqTextBox"),
            };
            string jsonPayload = JsonConvert.SerializeObject(simParams);
            var payload = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
            
            await ExecuteBackendProcess("/start-simulation", "simulation", payload);
        }
        catch (Exception ex)
        {
            AppendLog($"❌ UI ERROR: {ex.Message}");
        }
    }

    private async void StartBatchSimulationButton_Click(object sender, RoutedEventArgs e)
    {
        try
        {
             var batchParams = new BatchSimulationParams
            {
                R = ParseValue<float>("BatchRTextBox"),
                C = ParseValue<float>("BatchCTextBox"),
                time_sim = ParseValue<float>("BatchTimeSimTextBox"),
                amplitude = ParseValue<float>("BatchAmplitudeTextBox"),
                duty_start = ParseValue<float>("DutyStartTextBox"),
                duty_end = ParseValue<float>("DutyEndTextBox"),
                duty_steps = ParseValue<int>("DutyStepsTextBox"),
                freq_start = ParseValue<float>("FreqStartTextBox"),
                freq_end = ParseValue<float>("FreqEndTextBox"),
                freq_steps = ParseValue<int>("FreqStepsTextBox"),
            };
            string jsonPayload = JsonConvert.SerializeObject(batchParams);
            var payload = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

            await ExecuteBackendProcess("/start-simulation-batch", "simulation", payload);
        }
        catch (Exception ex)
        {
            AppendLog($"❌ UI ERROR: {ex.Message}");
        }
    }

    // --- Event Handler สำหรับปุ่ม Scaling ---
    private async void StartScalingButton_Click(object sender, RoutedEventArgs e)
    {
        await ExecuteBackendProcess("/start-scaling", "scaling");
    }

    // --- ฟังก์ชันกลางสำหรับส่งคำสั่งและรอฟังผล ---
    private async Task ExecuteBackendProcess(string endpoint, string logType, HttpContent? payload = null)
    {
        AppendLog($"Sending command to endpoint: {endpoint}...");
        try
        {
            var response = await _httpClient.PostAsync(endpoint, payload);
            response.EnsureSuccessStatusCode();
            var responseBody = await response.Content.ReadAsStringAsync();
            AppendLog($"✅ Backend Response: {responseBody}");
        }
        catch (Exception ex)
        {
            AppendLog($"❌ CRITICAL ERROR sending command: {ex.Message}");
            return;
        }
        await ListenToLogs(logType);
    }
    
    // (ฟังก์ชัน ListenToLogs และ AppendLog เหมือนเดิมทุกประการ)
    private async Task ListenToLogs(string logType) { /* ... โค้ดเหมือนเดิม ... */ }
    private void AppendLog(string message) { /* ... โค้ดเหมือนเดิม ... */ }
}