// EnergyPlus, Copyright (c) 1996-2019, The Board of Trustees of the University of Illinois,
// The Regents of the University of California, through Lawrence Berkeley National Laboratory
// (subject to receipt of any required approvals from the U.S. Dept. of Energy), Oak Ridge
// National Laboratory, managed by UT-Battelle, Alliance for Sustainable Energy, LLC, and other
// contributors. All rights reserved.
//
// NOTICE: This Software was developed under funding from the U.S. Department of Energy and the
// U.S. Government consequently retains certain rights. As such, the U.S. Government has been
// granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable,
// worldwide license in the Software to reproduce, distribute copies to the public, prepare
// derivative works, and perform publicly and display publicly, and to permit others to do so.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted
// provided that the following conditions are met:
//
// (1) Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright notice, this list of
//     conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//
// (3) Neither the name of the University of California, Lawrence Berkeley National Laboratory,
//     the University of Illinois, U.S. Dept. of Energy nor the names of its contributors may be
//     used to endorse or promote products derived from this software without specific prior
//     written permission.
//
// (4) Use of EnergyPlus(TM) Name. If Licensee (i) distributes the software in stand-alone form
//     without changes from the version obtained under this License, or (ii) Licensee makes a
//     reference solely to the software portion of its product, Licensee must refer to the
//     software as "EnergyPlus version X" software, where "X" is the version number Licensee
//     obtained under this License and may not use a different name for the software. Except as
//     specifically required in this Section (4), Licensee shall not use in a company name, a
//     product name, in advertising, publicity, or other promotional activities any name, trade
//     name, trademark, logo, or other designation of "EnergyPlus", "E+", "e+" or confusingly
//     similar designation, without the U.S. Department of Energy's prior written consent.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

// C++ Headers
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

// ObjexxFCL Headers
#include <ObjexxFCL/Fmath.hh>
#include <ObjexxFCL/gio.hh>
#include <ObjexxFCL/string.functions.hh>

// EnergyPlus Headers
#include <CommandLineInterface.hh>
#include <ExtCtrl.hh>
#include <DataEnvironment.hh>
#include <DataGlobals.hh>
#include <DataHVACGlobals.hh>
#include <DataPrecisionGlobals.hh>
#include <General.hh>
#include <UtilityRoutines.hh>
#include <DisplayRoutines.hh>

namespace EnergyPlus {

namespace ExtCtrl {
	// Module containing the external control

	// MODULE INFORMATION:
	//       AUTHOR         Takao Moriyama, IBM Corporation
	//       DATE WRITTEN   December 2017
	//       MODIFIED       na
	//       RE-ENGINEERED  na

	// PURPOSE OF THIS MODULE:
	// This module provides a repository for suporting external control

	// METHODOLOGY EMPLOYED:
	// na

	// REFERENCES:
	// na

	// OTHER NOTES:
	// na

	// Data
	// MODULE PARAMETER DEFINITIONS:
	// For sending observation

	using DataGlobals::TimeStepZone;
	using DataHVACGlobals::TimeStepSys;

	int const CMD_OBS_INIT(0);
	int const NUM_OBSS(100);
	int const CMD_OBS_INDEX_LOW(1);
	int const CMD_OBS_INDEX_HIGH(NUM_OBSS);
	Real64 obss[NUM_OBSS];
	Real64 const OBS_DATA_NULL(-123.0);

	// For receiving action
	int const CMD_ACT_REQ(0);
	int const NUM_ACTS(100);
	int const CMD_ACT_INDEX_LOW (1);
	int const CMD_ACT_INDEX_HIGH (NUM_ACTS);
	Real64 acts[NUM_ACTS];
	Real64 const ACT_DATA_NULL(-456.0);

	std::string const blank_string;

	// MODULE VARIABLE DECLARATIONS:
	// na

	// MODULE VARIABLE DEFINITIONS:
	//std::string String;
	bool ReportErrors(true);

	// Object Data
	std::ifstream act_ifs;
	std::ofstream obs_ofs;
	char *act_filename;
	char *obs_filename;
	int act_seq = 0;
	int obs_seq = 0;

	// Subroutine Specifications for the Module

	// Functions

	int
	InitializeExtCtrlRoutines()
	{
		static bool firstCall (true);
		if (firstCall) {
			firstCall = false;
			//DisplayString("InitializeExtCtrlRoutine(): First call");

			if ((act_filename = getenv("ACT_PIPE_FILENAME")) == NULL) {
				ShowFatalError("InitializeExtCtrlActRoutines: Environment variable ACT_PIPE_FILENAME not specified");
				DisplayString("InitializeExtCtrlRoutine: ACT file not specified");
				return 1;
			}
			if ((obs_filename = getenv("OBS_PIPE_FILENAME")) == NULL) {
				ShowFatalError("InitializeExtCtrlActRoutines: Environment variable OBS_PIPE_FILENAME not specified");
				return 1;
			}
		}
		return 0;
	}

	std::string
	ExtCtrlRead()
	{
		if (!act_ifs.is_open()) {
			act_ifs.open(act_filename);
			act_ifs.rdbuf()->pubsetbuf(0, 0); // Making unbuffered
			if (!act_ifs.is_open()) {
				ShowFatalError("ExtCtrlRead: ACT file could not open");
				return "";
			}
			DisplayString("ExtCtrlRead: Opened ACT file: " + std::string(act_filename));
		}
		std::string line;
	  again:
		act_ifs >> line;
		size_t idx = line.find(",");
		if (idx == std::string::npos) {
			goto again;
		}
		std::string seq = line.substr(0, idx);
		std::string val = line.substr(idx + 1, std::string::npos);
		assert(act_seq == seq);
		act_seq++;
		return val;
	}

	void
	ExtCtrlWrite(std::string str)
	{
		if (!obs_ofs.is_open()) {
			obs_ofs.open(obs_filename);
			if (!obs_ofs.is_open()) {
				ShowFatalError("ExtCtrlWrite: InitializeExtCtrlRoutine: OBS file could not open");
				return;
			}
			DisplayString("ExtCtrlWrite: Opened OBS file: " + std::string(obs_filename));
		}
		obs_ofs << obs_seq << "," << str << std::endl;
		obs_seq++;
	}

	void
	ExtCtrlFlush()
	{
		obs_ofs << "DELIMITER" << std::endl;
		obs_ofs.flush();
	}

	void
	InitializeObsData()
	{
		for (int i = 0; i < NUM_OBSS; i++)
			obss[i] = OBS_DATA_NULL;
	}

	void
	InitializeActData()
	{
		for (int i = 0; i < NUM_ACTS; i++)
			acts[i] = ACT_DATA_NULL;
	}

	Real64
	ExtCtrlObs(
		Real64 const cmd, // command code
		Real64 const arg  // command value
	)
	{
		Int64 cmdInt = cmd;

		if (InitializeExtCtrlRoutines()) {
			return -1.0;
		}
		if (cmdInt >= CMD_OBS_INDEX_LOW && cmdInt <= CMD_OBS_INDEX_HIGH) {
			//DisplayString("ExtCtrlObs: set obs[" + std::to_string(cmdInt) + "] = " + std::to_string(arg));
			obss[cmdInt - 1] = arg;
			return 0.0;
		}
		else if (cmdInt == CMD_OBS_INIT) {
			//DisplayString("ExtCtrlObs: INIT");
			// If not connected to the server, try to connect.
			// TODO:
			//ShowFatalError("Failed to connect to external service");
			return 0.0;
		}
		// TODO: Show error code
		ShowWarningMessage("Obs index " + std::to_string(cmdInt) + " is out of range [" + std::to_string(CMD_OBS_INDEX_LOW) + "..." + std::to_string(CMD_OBS_INDEX_HIGH) + "]");
		return -1.0;
	}

	Real64
	ExtCtrlAct(
		Real64 const cmd, // command code
		Real64 const arg  // command value
	)
	{
		Int64 cmdInt = cmd;
		Int64 argInt = arg;

		if (InitializeExtCtrlRoutines()) {
			return -1.0;
		}
		if (cmdInt >= CMD_ACT_INDEX_LOW && cmdInt <= CMD_ACT_INDEX_HIGH) {
			//DisplayString("ExtCtrlAct: get acts[" + std::to_string(cmdInt) + "] = " + std::to_string(acts[cmdInt - 1]));
			return acts[cmdInt - 1];
		}
		else if (cmdInt == CMD_ACT_REQ) {
			if (!(argInt >= 0 && argInt <= CMD_ACT_INDEX_HIGH)) {
				ShowWarningMessage("ExtCtrlAct: Number of obss " + std::to_string(argInt) + " must be in range [0..." + std::to_string(CMD_ACT_INDEX_HIGH) + "]");
				return -1.0;
			}
			// skip system timestep
			if (TimeStepSys < TimeStepZone) {
				return 0.0;
			}

			// Send observation data to the server, and receive next action.
			ExtCtrlWrite(std::to_string(argInt));
			for (int i = CMD_ACT_INDEX_LOW; i <= argInt; i++) {
				ExtCtrlWrite(std::to_string(obss[i - 1]));
			}
			ExtCtrlFlush();

			// Get action data
			std::string line;
			line = ExtCtrlRead();
			int NumActsReceived = std::stoi(line);
			assert(NumActsReceived >= 0 && MumActsReceived <= CMD_ACT_INDEX_HIGH);
			for (int i = 1; i <= NumActsReceived; i++) {
				line = ExtCtrlRead();
				double val = std::stod(line);
				if (i <= CMD_ACT_INDEX_HIGH) {
					acts[i - 1] = val;
				}
			}

			return 0.0;
		}
		ShowWarningMessage("Act index "+ std::to_string(cmdInt) + " is out of range [" + std::to_string(CMD_ACT_INDEX_LOW) + " to " + std::to_string(CMD_ACT_INDEX_HIGH) + "]");
		return -1.0;
	}

} // ExtCtrl

} // EnergyPlus
