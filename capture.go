package main

import(
  "fmt"
  "os"
  "github.com/google/gopacket"
  "github.com/google/gopacket/pcap"
  "github.com/google/gopacket/layers"
	"github.com/google/gopacket/pcapgo"
  "time"
  //"os/exec"
)

/* Check the given arguments */
func checkForArguments() bool{
  argsWithoutProg := os.Args[1:]
  if(len(argsWithoutProg) > 1){
    return true
  } else {
    return false
  }
}

var (

    snapshotLen int32  = 1024
    promiscuous  bool   = false
    err          error
    timeout      time.Duration = 30 * time.Second
    handle       *pcap.Handle
    packetCount int = 0 
)
func printPacketInfo(packet gopacket.Packet) {
    
    fmt.Println("\n------------------------------------------------------------------------------------------------------\n")
    // Let's see if the packet is an ethernet packet
    ethernetLayer := packet.Layer(layers.LayerTypeEthernet)
    if ethernetLayer != nil {
        fmt.Println("Ethernet layer detected.")
        ethernetPacket, _ := ethernetLayer.(*layers.Ethernet)
        fmt.Println("Source MAC: ", ethernetPacket.SrcMAC)
        fmt.Println("Destination MAC: ", ethernetPacket.DstMAC)
        // Ethernet type is typically IPv4 but could be ARP or other
        fmt.Println("Ethernet type: ", ethernetPacket.EthernetType)
        fmt.Println()
    }

    // Let's see if the packet is IP (even though the ether type told us)
    ipLayer := packet.Layer(layers.LayerTypeIPv4)
    if ipLayer != nil {
        fmt.Println("IPv4 layer detected.")
        ip, _ := ipLayer.(*layers.IPv4)

        // IP layer variables:
        // Version (Either 4 or 6)
        // IHL (IP Header Length in 32-bit words)
        // TOS, Length, Id, Flags, FragOffset, TTL, Protocol (TCP?),
        // Checksum, SrcIP, DstIP
        fmt.Printf("From %s to %s\n", ip.SrcIP, ip.DstIP)
        fmt.Println("Protocol: ", ip.Protocol)
        fmt.Println()
    }

    // Let's see if the packet is TCP
    tcpLayer := packet.Layer(layers.LayerTypeTCP)
    if tcpLayer != nil {
        fmt.Println("TCP layer detected.")
        tcp, _ := tcpLayer.(*layers.TCP)

        // TCP layer variables:
        // SrcPort, DstPort, Seq, Ack, DataOffset, Window, Checksum, Urgent
        // Bool flags: FIN, SYN, RST, PSH, ACK, URG, ECE, CWR, NS
        fmt.Printf("From port %d to %d\n", tcp.SrcPort, tcp.DstPort)
        fmt.Println("Sequence number: ", tcp.Seq)
        fmt.Println()
    }

    // Iterate over all layers, printing out each layer type
    fmt.Println("All packet layers:")
    for _, layer := range packet.Layers() {
        fmt.Println("- ", layer.LayerType())
    }

    // Check for errors
    if err := packet.ErrorLayer(); err != nil {
        fmt.Println("Error decoding some part of the packet:", err)
    }
    fmt.Println("\n------------------------------------------------------------------------------------------------------\n")
}
func main(){
  if(!checkForArguments()){
    fmt.Println("[ERROR]: Give the correct argument.")
    return
  }
  // Retrieve the interface
  device := os.Args[1]  

  // Open output pcap file and write header 
  f := os.Stdout
	w := pcapgo.NewWriter(f)
	w.WriteFileHeader(1024, layers.LinkTypeEthernet)
	defer f.Close()
  
  //Open device
  handle,err = pcap.OpenLive(device,1024,promiscuous,timeout)

  if err != nil{
    fmt.Println(err)
    return
  }
  defer handle.Close()

  // Use the handle as a packet source to process all packets
  packetSource := gopacket.NewPacketSource(handle,handle.LinkType())
  for packet := range packetSource.Packets(){
    //printPacketInfo(packet) 
	  w.WritePacket(packet.Metadata().CaptureInfo, packet.Data())
    packetCount++
    
    if packetCount > 10000{
      break
    }
  }  
}
