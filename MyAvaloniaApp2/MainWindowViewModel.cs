// MyAvaloniaApp2/MainWindowViewModel.cs
using Avalonia.Threading;
using Newtonsoft.Json;
using System;
using System.Globalization;
using System.Net.Http;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Collections.Generic; // <--- เพิ่ม using statement นี้เข้าไปด้วย

namespace MyAvaloniaApp2;

// Model สำหรับส่งข้อมูลไปที่ Backend API
public class SimulationRequest
{
    public float R { get; set; }
    public float C { get; set; }
    public float dt { get; set; }
    public float setpoint_level { get; set; }
    public float time_sim { get; set; }
    public float amplitude { get; set; }
    public string control_mode { get; set; } = "voltage";
    public string signal_type { get; set; } = "pwm";
    public float? duty { get; set; }
    public float? freq { get; set; }
    public float? duty_start { get; set; }
    public float? duty_end { get; set; }
    public int? duty_steps { get; set; }
    public float? freq_start { get; set; }
    public float? freq_end { get; set; }
    public int? freq_steps { get; set; }
}

public class MainWindowViewModel : ViewModelBase
{
    private static readonly HttpClient _httpClient = new() { BaseAddress = new Uri("http://127.0.0.1:8000") };
    private ClientWebSocket? _webSocket;
    private CancellationTokenSource? _cancellationTokenSource;

    #region Properties for UI Binding
    private string _r = "1.5";
    public string R { get => _r; set { _r = value; OnPropertyChanged(); } }
    private string _c = "2.0";
    public string C { get => _c; set { _c = value; OnPropertyChanged(); } }
    private string _dt = "0.01";
    public string Dt { get => _dt; set { _dt = value; OnPropertyChanged(); } }
    private string _setpoint = "5.0";
    public string Setpoint { get => _setpoint; set { _setpoint = value; OnPropertyChanged(); } }
    private string _timeSim = "30.0";
    public string TimeSim { get => _timeSim; set { _timeSim = value; OnPropertyChanged(); } }
    private string _amplitude = "12.0";
    public string Amplitude { get => _amplitude; set { _amplitude = value; OnPropertyChanged(); } }
    private string _duty = "0.5";
    public string Duty { get => _duty; set { _duty = value; OnPropertyChanged(); } }
    private string _freq = "1.0";
    public string Freq { get => _freq; set { _freq = value; OnPropertyChanged(); } }
    private string _dutyStart = "0.1";
    public string DutyStart { get => _dutyStart; set { _dutyStart = value; OnPropertyChanged(); } }
    private string _dutyEnd = "1.0";
    public string DutyEnd { get => _dutyEnd; set { _dutyEnd = value; OnPropertyChanged(); } }
    private string _dutySteps = "10";
    public string DutySteps { get => _dutySteps; set { _dutySteps = value; OnPropertyChanged(); } }
    private string _freqStart = "0.1";
    public string FreqStart { get => _freqStart; set { _freqStart = value; OnPropertyChanged(); } }
    private string _freqEnd = "2.0";
    public string FreqEnd { get => _freqEnd; set { _freqEnd = value; OnPropertyChanged(); } }
    private string _freqSteps = "5";
    public string FreqSteps { get => _freqSteps; set { _freqSteps = value; OnPropertyChanged(); } }
    private bool _isBatchModeEnabled;
    public bool IsBatchModeEnabled { get => _isBatchModeEnabled; set { _isBatchModeEnabled = value; OnPropertyChanged(); } }
    private string _logText = "System Ready. Waiting for command.";
    public string LogText { get => _logText; set { _logText = value; OnPropertyChanged(); } }
    private bool _isBusy = false;
    public bool IsBusy
    {
        get => _isBusy;
        set
        {
            _isBusy = value;
            OnPropertyChanged();
            OnPropertyChanged(nameof(IsNotBusy));

        }
    }
    #endregion
    
     #region ComboBox Properties
    public List<string> ControlModes { get; } = new() { "voltage", "current" };
    
    private string _selectedControlMode = "voltage";
    public string SelectedControlMode { get => _selectedControlMode; set { _selectedControlMode = value; OnPropertyChanged(); } }

    public List<string> SignalTypes { get; } = new() { "pwm", "step", "ramp", "impulse" };

    private string _selectedSignalType = "pwm";
    public string SelectedSignalType { get => _selectedSignalType; set { _selectedSignalType = value; OnPropertyChanged(); } }
    #endregion



    public bool IsNotBusy => !IsBusy; 

    public ICommand RunSimulationCommand { get; }
    public ICommand StartScalingCommand { get; }

    public MainWindowViewModel()
{
    // นำเงื่อนไข CanExecute (_ => !IsBusy) กลับมาเหมือนเดิม
    RunSimulationCommand = new RelayCommand(async _ => await ExecuteRunSimulation()); // <== ต้องเป็นแบบนี้!
    StartScalingCommand = new RelayCommand(async _ => await ExecuteStartScaling()); // <== ต้องเป็นแบบนี้!

}


    private async Task ExecuteRunSimulation()
    {
        if (IsBusy) return; // ป้องกันการเรียกซ้ำขณะกำลังทำงาน
        IsBusy = true;
        try
        {
            var request = new SimulationRequest {
                R = float.Parse(R, CultureInfo.InvariantCulture),
                C = float.Parse(C, CultureInfo.InvariantCulture),
                dt = float.Parse(Dt, CultureInfo.InvariantCulture),
                setpoint_level = float.Parse(Setpoint, CultureInfo.InvariantCulture),
                time_sim = float.Parse(TimeSim, CultureInfo.InvariantCulture),
                amplitude = float.Parse(Amplitude, CultureInfo.InvariantCulture),
                control_mode = SelectedControlMode,
                signal_type = SelectedSignalType
            };

            if (IsBatchModeEnabled)
            {
                request.duty_start = float.Parse(DutyStart, CultureInfo.InvariantCulture);
                request.duty_end = float.Parse(DutyEnd, CultureInfo.InvariantCulture);
                request.duty_steps = int.Parse(DutySteps, CultureInfo.InvariantCulture);
                request.freq_start = float.Parse(FreqStart, CultureInfo.InvariantCulture);
                request.freq_end = float.Parse(FreqEnd, CultureInfo.InvariantCulture);
                request.freq_steps = int.Parse(FreqSteps, CultureInfo.InvariantCulture);
            }
            else
            {
                request.duty = float.Parse(Duty, CultureInfo.InvariantCulture);
                request.freq = float.Parse(Freq, CultureInfo.InvariantCulture);
            }
            
            string jsonPayload = JsonConvert.SerializeObject(request, new JsonSerializerSettings { NullValueHandling = NullValueHandling.Ignore });
            var payload = new StringContent(jsonPayload, Encoding.UTF8, "application/json");

            await ExecuteBackendProcess("/simulation/run", "simulation", payload);
        }
        catch (Exception ex)
        {
            AppendLog($"❌ UI ERROR: {ex.Message}");
        }
        finally
        {
            IsBusy = false;
        }
    }

    private async Task ExecuteStartScaling()
    {
        if (IsBusy) return; // ป้องกันการเรียกซ้ำขณะกำลังทำงาน
        IsBusy = true;
            try
        {
            // แก้ไข: ต้องมี try...finally ที่นี่ด้วยถึงจะสมบูรณ์แบบ!
            await ExecuteBackendProcess("/start-scaling", "scaling");
        }
        finally
        {
            IsBusy = false;
        }
        
    }

    private async Task ExecuteBackendProcess(string endpoint, string logType, HttpContent? payload = null)
    {
        AppendLog($"Sending command to endpoint: {endpoint}...");
        try
        {
            var response = await _httpClient.PostAsync(endpoint, payload);
            response.EnsureSuccessStatusCode();
            var responseBody = await response.Content.ReadAsStringAsync();
            AppendLog($"✅ Backend Response: {responseBody}");
             _ = ListenToLogs(logType); 
        }
        catch (Exception ex)
        {
            AppendLog($"❌ CRITICAL ERROR sending command: {ex.Message}");
        }
    }
    
    private void AppendLog(string message)
    {
        Dispatcher.UIThread.InvokeAsync(() => { LogText += $"\n{message}"; });
    }

    private async Task ListenToLogs(string logType)
    {
        if (_webSocket?.State == WebSocketState.Open) {
            await _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
        }
        _cancellationTokenSource = new CancellationTokenSource();
        _webSocket = new ClientWebSocket();
        try
        {
            Uri wsUri = new UriBuilder(_httpClient.BaseAddress!) { Scheme = "ws", Path = "ws/log" }.Uri;
            await _webSocket.ConnectAsync(wsUri, _cancellationTokenSource.Token);
            AppendLog($"[WebSocket] Connected. Listening for '{logType}' logs...");
            
            var messageBuffer = Encoding.UTF8.GetBytes(logType);
            await _webSocket.SendAsync(new ArraySegment<byte>(messageBuffer), WebSocketMessageType.Text, true, _cancellationTokenSource.Token);

            var receiveBuffer = new byte[1024];
            while (_webSocket.State == WebSocketState.Open && !_cancellationTokenSource.Token.IsCancellationRequested)
            {
                var result = await _webSocket.ReceiveAsync(new ArraySegment<byte>(receiveBuffer), _cancellationTokenSource.Token);
                if (result.MessageType == WebSocketMessageType.Close) {
                    await _webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, string.Empty, CancellationToken.None);
                    AppendLog("[WebSocket] Connection closed by server.");
                } else {
                    var receivedMessage = Encoding.UTF8.GetString(receiveBuffer, 0, result.Count);
                    AppendLog(receivedMessage);
                }
            }
        }
        catch (Exception ex)
        {
            AppendLog($"[WebSocket] ERROR: {ex.Message}");
        }
    }
}