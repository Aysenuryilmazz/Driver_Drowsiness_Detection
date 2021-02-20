import React, { Component } from "react";
import { MDBTable, MDBTableBody, MDBTableHead } from "mdbreact";
import io from "socket.io-client";

let endPoint = "http://127.0.0.1:5000";
let socket = io.connect(`${endPoint}`);

class Home extends Component {
  state = {
    messages: {},
    message: "",
    counter: 0,
    isDrowsy1: false,
    isDrowsy2: false
  };

  componentDidMount = async () => {
    if (this.state.counter === 0) {
      socket.emit("message", "message");
      await this.setState({ counter: 1 });
    }
    socket.on("message", (msg) => {
      console.log(msg.EARLYDETECTION)
      console.log(typeof msg.EARLYDETECTION)
      this.setState({
        messages: msg,
      });

      if (String(msg.EARLYDETECTION) !== "CALIB"){
        if(Number(msg.EARLYDETECTION) < 0.53 ){
          this.setState({isDrowsy2: true})
        }else{
          this.setState({isDrowsy2: false})
        }
      }
      if (String(msg.DROWSINESS) !== "CALIB"){
        if(Number(msg.DROWSINESS) == 1 ){
          this.setState({isDrowsy1: true})
        }else{
          this.setState({isDrowsy1: true})
        }
      }
    });

    console.log(this.state.messages);
  };
  

  render() {
    return (
      <div className="app">
        <div className="drowsiness">
          <h2 className="status">
            <span style={{verticalAlign: "middle"}} >DROWSINESS STATUS: </span><span style={{verticalAlign: "middle"}} className={this.state.isDrowsy1 === true ? "drowsiness-text" : "awake-text"}>{String(this.state.messages.DROWSINESS) !== "CALIB" ? this.state.isDrowsy1 ? "DROWSY": "AWAKE" : "CALIBRATING..."}</span>
          </h2>
          <h2 className="status">
            <span style={{verticalAlign: "middle"}} >EARLY DETECTION: </span><span style={{verticalAlign: "middle"}} className={this.state.isDrowsy2 === true ? "drowsiness-text" : "awake-text"}>{String(this.state.messages.EARLYDETECTION) !== "CALIB" ? this.state.isDrowsy2 ? "WARNING": "OK.": "CALIBRATING..."}</span>
          </h2>
        </div>

        <div className="main">
          <div>
            <img
              src={"http://127.0.0.1:5000/video_feed"}
              alt="logo"
              className="video-frame"
            />
          </div>

          <div className="info">
            <div className="tables">
              <MDBTable className="info-table" hover>
                <thead>
                  <tr>
                    <th></th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody className="table-body">
                  <tr>
                    <td>Eye Aspect Ratio</td>
                    <td>{Number(this.state.messages.EAR).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Percantage of Eye Closure</td>
                    <td>{Number(this.state.messages.PERCLOS).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Mouth Aspect Ratio</td>
                    <td>{Number(this.state.messages.MAR).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Mouth Over Eye</td>
                    <td>{Number(this.state.messages.MOE).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Eye Circularity</td>
                    <td>{Number(this.state.messages.EC).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Level of Brow</td>
                    <td>{Number(this.state.messages.LEB).toFixed(2)}</td>
                  </tr>
                  <tr>
                    <td>Size of Pupil</td>
                    <td>{Number(this.state.messages.SOP).toFixed(2)}</td>
                  </tr>
                </tbody>
              </MDBTable>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default Home;