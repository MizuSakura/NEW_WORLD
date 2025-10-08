using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Threading;
using System;
using System.Net.Http;
using System.Net.Http.Json; // <-- เครื่องมือใหม่!
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json; // <-- เครื่องมือแพ็ค JSON ที่เราเพิ่งติดตั้ง!

namespace MyAvaloniaApp2;

// สร้าง Class เพื่อเก็บข้อมูล "กล่องพัสดุ" ของเรา
public class SimulationParams
{
    // ชื่อตัวแปรต้องตรงกับใน Python Pydantic Model เป๊ะๆ!
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
    // (ค่าอื่นๆ ที่มี default ใน Python เราไม่จำเป็นต้องใส่ที่นี่ก็ได้)
    
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

    private async void StartProcessButton_Click(object sender, RoutedEventArgs e)
    {
        var button = (Button)sender;
        string endpoint = "";
        string logType = "";
        HttpContent? payload = null; // <-- ตัวแปรสำหรับเก็บ "กล่องพัสดุ"

        switch (button.Name)
        {
            case "StartSimulationButton":
                endpoint = "/start-simulation";
                logType = "simulation";
                // --- หัวใจของการแก้ไข! ---
                // รวบรวมข้อมูลจาก "แผงควบคุม"
                var simParams = new SimulationParams
                {
                    R = float.Parse(this.FindControl<TextBox>("RTextBox").Text),
                    C = float.Parse(this.FindControl<TextBox>("CTextBox").Text),
                    dt = float.Parse(this.FindControl<TextBox>("DtTextBox").Text),
                    setpoint_level = float.Parse(this.FindControl<TextBox>("SetpointTextBox").Text),
                    control_mode = (this.FindControl<ComboBox>("ControlModeComboBox").SelectedItem as ComboBoxItem).Content.ToString(),
                    time_sim = float.Parse(this.FindControl<TextBox>("TimeSimTextBox").Text),
                    signal_type = (this.FindControl<ComboBox>("SignalTypeComboBox").SelectedItem as ComboBoxItem).Content.ToString(),
                    amplitude = float.Parse(this.FindControl<TextBox>("AmplitudeTextBox").Text),
                    duty = float.Parse(this.FindControl<TextBox>("DutyTextBox").Text),
                    freq = float.Parse(this.FindControl<TextBox>("FreqTextBox").Text),
                };
                // "แพ็ค" ข้อมูลลงกล่อง JSON
                string jsonPayload = JsonConvert.SerializeObject(simParams);
                payload = new StringContent(jsonPayload, Encoding.UTF8, "application/json");
                break;

            case "StartScalingButton":
                endpoint = "/start-scaling";
                logType = "scaling";
                // Scaling ยังไม่ต้องส่งข้อมูลอะไรไป
                break;

            default:
                AppendLog("❌ Unknown button clicked.");
                return;
        }

        // 1. ส่งคำสั่ง POST (พร้อม "กล่องพัสดุ" ถ้ามี)
        AppendLog($"Sending command to endpoint: {endpoint}...");
        try
        {
            var response = await _httpClient.PostAsync(endpoint, payload); // <-- ส่ง payload ไปด้วย!
            response.EnsureSuccessStatusCode();
            var responseBody = await response.Content.ReadAsStringAsync();
            AppendLog($"✅ Backend Response: {responseBody}");
        }
        catch (Exception ex)
        {
            AppendLog($"❌ CRITICAL ERROR sending command: {ex.Message}");
            return;
        }

        // 2. "เปิดวิทยุ" เพื่อรอฟัง Log (เหมือนเดิม)
        await ListenToLogs(logType);
    }
    
    // (ฟังก์ชัน ListenToLogs และ AppendLog เหมือนเดิมทุกประการ)
    private async Task ListenToLogs(string logType)
    {
        if (_webSocket != null && _webSocket.State == WebSocketState.Open) { _cancellationTokenSource?.Cancel(); await _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Starting new log stream", CancellationToken.None); }
        _webSocket = new ClientWebSocket();
        _cancellationTokenSource = new CancellationTokenSource();
        AppendLog($"Opening WebSocket to listen for '{logType}' logs...");
        try
        {
            Uri wsUri = new Uri($"ws://127.0.0.1:8000/ws/log");
            await _webSocket.ConnectAsync(wsUri, _cancellationTokenSource.Token);
            AppendLog("WebSocket connected. Waiting for logs...");
            var messageBytes = Encoding.UTF8.GetBytes(logType);
            await _webSocket.SendAsync(new ArraySegment<byte>(messageBytes), WebSocketMessageType.Text, true, _cancellationTokenSource.Token);
            var buffer = new byte[4096];
            while (_webSocket.State == WebSocketState.Open && !_cancellationTokenSource.IsCancellationRequested)
            {
                var result = await _webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), _cancellationTokenSource.Token);
                if (result.MessageType == WebSocketMessageType.Text)
                {
                    var logMessage = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    Dispatcher.UIThread.Post(() => AppendLog(logMessage));
                }
            }
        }
        catch (Exception ex) { Dispatcher.UIThread.Post(() => AppendLog($"❌ WebSocket ERROR: {ex.Message}")); }
    }
    private void AppendLog(string message)
    {
        var logTextBlock = this.FindControl<TextBlock>("LogTextBlock");
        if (logTextBlock != null) { logTextBlock.Text += $"[{DateTime.Now:HH:mm:ss}] {message}\n"; var logScrollViewer = this.FindControl<ScrollViewer>("LogScrollViewer"); logScrollViewer?.ScrollToEnd(); }
    }
}