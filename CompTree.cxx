// _____________________________________________________________________________
// Standard includes
#include <set>
#include <string>
#include <vector>
#include <map>

// _____________________________________________________________________________
// EDM includes
#include "xAODEventInfo/EventInfo.h"
#include "xAODEventInfo/EventAuxInfo.h"
#include "xAODJet/Jet.h"
#include "xAODJet/JetAttributes.h"
#include "xAODJet/JetContainer.h"
#include "xAODRootAccess/tools/Message.h"
#include "xAODEventInfo/EventInfo.h"
#include "xAODEventInfo/EventAuxInfo.h"
#include "xAODTruth/TruthParticle.h"
#include "xAODTruth/TruthParticleAuxContainer.h"
#include <xAODTruth/TruthParticleContainer.h> 
#include "xAODPFlow/TrackCaloClusterContainer.h"
#include "xAODParticleEvent/IParticleLink.h"
#include "xAODParticleEvent/IParticleLinkContainer.h"
#include "xAODTracking/TrackParticleAuxContainer.h"
#include "xAODTracking/TrackParticleContainer.h"
#include "xAODCaloEvent/CaloClusterContainer.h"
#include "xAODCaloEvent/CaloCluster.h"
#include "xAODCore/ShallowAuxContainer.h" 
#include "xAODCore/ShallowCopy.h"
#include "JetCalibTools/JetCalibrationTool.h"
#include "JetCalibTools/IJetCalibrationTool.h"
#include "AsgTools/AnaToolHandle.h"


// _____________________________________________________________________________
// Athena includes
#include "AthContainers/ConstDataVector.h"
#include "AthLinks/ElementLink.h"

// _____________________________________________________________________________
// ROOT includes
#include "TLorentzVector.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TH1F.h"
#include "TSystem.h"
 
#include "GaudiKernel/ITHistSvc.h"

#include "CompTree.h"

using CLHEP::GeV;
using xAOD::IParticle;

CompTree::CompTree( const std::string& name, ISvcLocator* pSvcLocator ) : 
  AthAlgorithm( name, pSvcLocator ),
  m_tHistSvc(nullptr),
  m_stream("/PolComponents/")
{
  declareProperty("MyStream", m_stream);
  declareProperty("ApplyCalibration"            , m_applyCalibration   = false);
  declareProperty("ApplyTruthMatching"          , m_applyTruthMatching = true);
  declareProperty("LongitudinalPolarization"    , m_isSigLong  = false);
  declareProperty("TransversePolarization"      , m_isSigTrans = false);
  declareProperty("maxTrkJetDR"                 , m_maxJetDR = 0.75);
  declareProperty("maxTrkJetDR"                 , m_maxJetDR = 0.75);
  declareProperty("maxEta"                      , m_maxEta   = 2.0);
  declareProperty("minPt"                       , m_minPt    = 200.);
  declareProperty("minMass"                     , m_minMass  = 50.);
  declareProperty("maxMass"                     , m_maxMass  = 150.);
  declareProperty("JZslice"                     , m_JZslice  = "Pol");
  //declareProperty("eventsFile"                  , m_eventsFile  = "");
  //declareProperty("eventsTree"                  , m_eventsTree  = "events_training");
}


CompTree::~CompTree() {}


StatusCode CompTree::initialize() {
  ATH_MSG_INFO ("Initializing " << name() << "...");
  
  if (service("THistSvc",m_tHistSvc).isFailure()) {
    ATH_MSG_ERROR( "initialize() Could not find Hist Service!" );
    return StatusCode::FAILURE;
  }


  TString UFOjetAlgo    = "AntiKt10UFOCSSKSoftDropBeta100Zcut10";  // Jet collection, for example AntiKt4EMTopo or AntiKt4LCTopo (see below)
  TString UFOconfig     = "JES_MC16recommendation_R10_UFO_CSSK_SoftDrop_JMS_01April2020.config";  // Global config (see below)
  TString UFOcalibSeq   = "EtaJES_JMS"; // Calibration sequence to apply (see below)
  TString UFOcalibArea  = "00-04-82"; // Calibration Area tag (see below)
  bool isData           =  false; // bool describing if the events are data or from simulation
    
  
  if (m_applyCalibration) {
    m_UFOjetCalibration.setTypeAndName("JetCalibrationTool/UFOname");
    if( !m_UFOjetCalibration.isUserConfigured() ){
      ATH_CHECK( m_UFOjetCalibration.setProperty("JetCollection",UFOjetAlgo.Data()) );
      ATH_CHECK( m_UFOjetCalibration.setProperty("ConfigFile",UFOconfig.Data()) );
      ATH_CHECK( m_UFOjetCalibration.setProperty("CalibSequence",UFOcalibSeq.Data()) );
      ATH_CHECK( m_UFOjetCalibration.setProperty("CalibArea",UFOcalibArea.Data()) );
      ATH_CHECK( m_UFOjetCalibration.setProperty("IsData",isData) );
      ATH_CHECK( m_UFOjetCalibration.retrieve() );
    }
  }

  ITHistSvc* thistSvc = 0;
  if (service("THistSvc",thistSvc).isFailure()) {
    ATH_MSG_ERROR( "initialize() Could not find Hist Service!" );
  return StatusCode::FAILURE;}

  if(bookTree("UFO").isFailure())
    return StatusCode::FAILURE;

  // creating the tree for the weights
  std::string wtreename = "totWeights";
  TTree * totWeights = new TTree (wtreename.c_str(), "ntuple");   
  totWeights->Branch("Weight"          , &m_Weight           );    
  totWeights->Branch("TruthLeadingZTheta" , &m_theta_Z_0      );    
  totWeights->Branch("TruthLeadingZJetTheta"   , &m_theta_Zjet_0   );    
  totWeights->Branch("TruthSubLeadingZTheta" , &m_theta_Z_1      );    
  totWeights->Branch("TruthSubLeadingZJetTheta"   , &m_theta_Zjet_1   );    
  totWeights->Branch("TruthLeadingWTheta" , &m_theta_W_0      );    
  totWeights->Branch("TruthLeadingWJetTheta"   , &m_theta_Wjet_0   );    
  totWeights->Branch("TruthSubLeadingWTheta" , &m_theta_W_1      );    
  totWeights->Branch("TruthSubLeadingWJetTheta"   , &m_theta_Wjet_1   );    
  m_trees.insert(std::pair<std::string, TTree*>(wtreename, totWeights));
  if (m_tHistSvc) {
    if((m_tHistSvc->regTree(m_stream+wtreename, m_trees.at(wtreename))).isFailure()) {
      ATH_MSG_ERROR( "initialize() Could not register the validation Tree!" );
      return StatusCode::FAILURE;
    }
  }
  
  m_Weight = -99999;
  m_theta_Z_0 = -99999;
  m_theta_Zjet_0 = -99999;
  m_theta_Z_1 = -99999;
  m_theta_Zjet_1 = -99999;
  m_theta_W_0 = -99999;
  m_theta_Wjet_0 = -99999;
  m_theta_W_1 = -99999;
  m_theta_Wjet_1 = -99999;

  //retrieve file containing what events to keep
  /*TFile *eventfile = TFile::Open(m_eventsFile.c_str(), "READ");
  std::set<int>* eventvector;
  eventfile->GetObject(m_eventsTree.c_str(), eventvector);
  m_eventvector = *eventvector;*/
  return StatusCode::SUCCESS;
}


StatusCode CompTree::execute() {  
  ATH_MSG_DEBUG ("Executing " << name() << "...");
 


  const xAOD::EventInfo* info = nullptr;
  if (evtStore()->retrieve(info).isFailure()){
    ATH_MSG_FATAL( "Unable to retrieve Event Info" );
    return StatusCode::FAILURE;
  }


  const auto truths = getContainer<xAOD::JetContainer>("AntiKt10TruthSoftDropBeta100Zcut10Jets");  
  if (not truths) return StatusCode::FAILURE;

  const auto all_tracks = getContainer<xAOD::TrackParticleContainer>("InDetTrackParticles");  
  if (not all_tracks) return StatusCode::FAILURE;
    
  const auto truthParticles = getContainer<xAOD::TruthParticleContainer>("TruthParticles");  
  if (not truthParticles) return StatusCode::FAILURE;
  
  const auto truthParticlesWZ = getContainer<xAOD::TruthParticleContainer>("TruthBosonsWithDecayParticles");  
  if (not truthParticlesWZ) return StatusCode::FAILURE;

  const auto UFOjets_beforeCalib = getContainer<xAOD::JetContainer>("AntiKt10UFOCSSKSoftDropBeta100Zcut10Jets");  
  if (not UFOjets_beforeCalib) return StatusCode::FAILURE;
  
  std::string UFOname = "AntiKt10UFOCSSKSoftDropBeta100Zcut10Jets"; 

  const xAOD::JetContainer* UFOjets = UFOjets_beforeCalib;
  if (m_applyCalibration) {
    UFOjets = calibrateAndRecordShallowCopyJetCollection(UFOjets_beforeCalib, UFOname);
    if(!UFOjets){
      ATH_MSG_WARNING(  "Unable to create calibrated UFO jet shallow copy container" );
      return StatusCode::SUCCESS;
    }
    
  }


  std::map<std::string, float > weight_mult;
  weight_mult["JZ3W"]= 520148668.1576591;
  weight_mult["JZ4W"]= 183502636.73724174;
  weight_mult["JZ5W"]= 95273528.99611431;
  weight_mult["JZ6W"]= 21089952.986331422;
  weight_mult["JZ7W"]= 5647046.150208907;
  weight_mult["Pol"]= 1;

  //m_eventNumber = info->eventNumber();
  //if(m_eventvector.find(m_eventNumber)!= m_eventvector.end() or m_eventvector.find(-1*m_eventNumber)!= m_eventvector.end()) {

  //___________________________________________UFO jets__________________________________________________________
  // evaluate the leadings in mass of the leadings is pt
  std::vector<const xAOD::Jet*> UFOleadings = {nullptr, nullptr};

  std::vector<const xAOD::Jet*> tmp_UFOleadings;
  if (UFOjets->size()>0)
    tmp_UFOleadings.push_back(UFOjets->at(0));
  if (UFOjets->size()>1)
    tmp_UFOleadings.push_back(UFOjets->at(1));
  
  // fill the leadings jets if they satisfy the eta and pt requirements
  if (tmp_UFOleadings.size()>0 
  and fabs(tmp_UFOleadings.at(0)->eta())<m_maxEta 
  and tmp_UFOleadings.at(0)->pt()/1e3>m_minPt)
    UFOleadings.at(0) = tmp_UFOleadings.at(0);
  
  if (tmp_UFOleadings.size()>1 
  and fabs(tmp_UFOleadings.at(1)->eta())<m_maxEta
  and tmp_UFOleadings.at(1)->pt()/1e3>m_minPt)
    UFOleadings.at(1) = tmp_UFOleadings.at(1);
  
  std::vector<const xAOD::Jet*>             UFOtruth_matches = {nullptr, nullptr};
  std::vector<const xAOD::TruthParticle*> UFOtruth_particles = {nullptr, nullptr};

  
  int UFOpos = 0;
  for (const auto& jet: UFOleadings) {
    UFOpos++;
    if (not jet) continue;

    //____________FIND THE MATCHING TRUTH JETS_______________
    auto truth_matched = ClusterMatched<xAOD::JetContainer,xAOD::Jet>(jet,truths);
    // apply mass requirement on the truth jet once you have matched
    if (not truth_matched or (truth_matched->m()/1e3<m_minMass or truth_matched->m()/1e3>m_maxMass))
      continue;
    
    UFOtruth_matches.at(UFOpos-1) = truth_matched;
    if (m_applyTruthMatching and truth_matched and truthParticlesWZ and (m_isSigLong or m_isSigTrans))
        UFOtruth_particles.at(UFOpos-1) = ClusterMatched<xAOD::TruthParticleContainer,xAOD::TruthParticle>(truth_matched,truthParticlesWZ);
    if (m_applyTruthMatching and truth_matched and truthParticles and !(m_isSigLong or m_isSigTrans))
        UFOtruth_particles.at(UFOpos-1) = ClusterMatched<xAOD::TruthParticleContainer,xAOD::TruthParticle>(truth_matched,truthParticles);
  
  }



  //if(m_eventvector.find(m_eventNumber)!= m_eventvector.end()){
  if(UFOleadings.at(0)!=NULL && UFOtruth_matches.at(0)!=NULL && UFOtruth_particles.at(0)!=NULL){
    
    m_eventNumber = info->eventNumber();
    m_eventWeight = info->mcEventWeight()*weight_mult[m_JZslice.c_str()];

    m_eta                   = UFOleadings.at(0)->eta();
    m_phi                   = UFOleadings.at(0)->phi();
    m_E                     = UFOleadings.at(0)->e()/1e3;
    m_m                     = UFOleadings.at(0)->m()/1e3;
    m_pt                    = UFOleadings.at(0)->pt()/1e3;
    std::vector<xAOD::JetConstituent> JconstWrapped;
    try{JconstWrapped = UFOleadings.at(0)->getConstituents().asSTLVector();}
    catch (std::exception e){ std::cout << "WEIRD" << std::endl; }

    int Max = (JconstWrapped.size() < m_nComp) ? JconstWrapped.size() : m_nComp; 
    for (size_t iConst = 0; iConst < Max; ++iConst){
      /*if(JconstWrapped.at(iConst)->rawConstituent()->pt()<2)
        continue;*/
      int sign= (m_eta==0) ? 1 : (m_eta/fabs(m_eta)); 
      m_comp_eta[iConst] = (JconstWrapped.at(iConst)->rawConstituent()->eta()-m_eta)*sign;
      m_comp_phi[iConst] = UFOleadings.at(0)->p4().DeltaPhi(JconstWrapped.at(iConst)->rawConstituent()->p4());
      m_comp_delta[iConst] = UFOleadings.at(0)->p4().DeltaR(JconstWrapped.at(iConst)->rawConstituent()->p4());
      m_comp_pt[iConst] = log(JconstWrapped.at(iConst)->rawConstituent()->pt());
      m_comp_E[iConst] = log(JconstWrapped.at(iConst)->rawConstituent()->e());
      m_comp_pt_frac[iConst] = log(JconstWrapped.at(iConst)->rawConstituent()->pt()/UFOleadings.at(0)->pt());
      m_comp_E_frac[iConst] = log(JconstWrapped.at(iConst)->rawConstituent()->e()/UFOleadings.at(0)->e());
      const xAOD::TrackCaloCluster* ufo = dynamic_cast<const xAOD::TrackCaloCluster*>(JconstWrapped.at(iConst)->rawConstituent());
      m_comp_taste[iConst] = ufo->taste();
    }
    //___________HERE WE DEFINE OUR LABELS ! NEED TO BE EXTRA CAREFUL__________________
    m_truth_isW          = (UFOtruth_particles.at(0)->isW() && UFOtruth_matches.at(0)->m()/GeV > 50 && UFOtruth_matches.at(0)->m()/GeV < 100);
    m_truth_isZ          = (UFOtruth_particles.at(0)->isZ() && UFOtruth_matches.at(0)->m()/GeV > 60 && UFOtruth_matches.at(0)->m()/GeV < 100);
    m_truth_isV          = m_truth_isW || m_truth_isZ;
    m_truth_isQCD        = !m_truth_isV;
    if(m_truth_isV){
      m_truth_isLong       = m_isSigLong;
      m_truth_isTrans      = m_isSigTrans;
    }
    m_truth_isW_Long     = m_truth_isW && m_truth_isLong; 
    m_truth_isZ_Long     = m_truth_isZ && m_truth_isLong; 
    m_truth_isW_Trans    = m_truth_isW && m_truth_isTrans;
    m_truth_isZ_Trans    = m_truth_isZ && m_truth_isTrans;

    m_isLead                       = true; 
    writeTree("UFO");
  }   
  else 
    std::cout << "Problematic lead event" << std::endl;
  //}
  //if(m_eventvector.find(-1*m_eventNumber)!= m_eventvector.end()){
  if(UFOleadings.at(1)!=NULL && UFOtruth_matches.at(1)!=NULL && UFOtruth_particles.at(1)!=NULL){
    
    m_eventNumber = info->eventNumber();
    m_eventWeight = info->mcEventWeight()*weight_mult[m_JZslice.c_str()];

    m_eta                   = UFOleadings.at(1)->eta();
    m_phi                   = UFOleadings.at(1)->phi();
    m_E                     = UFOleadings.at(1)->e()/1e3;
    m_m                     = UFOleadings.at(1)->m()/1e3;
    m_pt                    = UFOleadings.at(1)->pt()/1e3;
    std::vector<xAOD::JetConstituent> JconstWrapped;
    try{JconstWrapped = UFOleadings.at(1)->getConstituents().asSTLVector();}
    catch (std::exception e){ std::cout << "WEIRD" << std::endl; }

    int Max = (JconstWrapped.size() < 100) ? JconstWrapped.size() : 100; 
    for (size_t iConst = 0; iConst < Max; ++iConst){
      /*if(JconstWrapped.at(iConst)->rawConstituent()->pt()<2)
        continue;*/
      int sign= (m_eta==0) ? 1 : (m_eta/fabs(m_eta)); 
      m_comp_eta[iConst] = (JconstWrapped.at(iConst)->rawConstituent()->eta()-m_eta)*sign;
      m_comp_phi[iConst] = UFOleadings.at(1)->p4().DeltaPhi(JconstWrapped.at(iConst)->rawConstituent()->p4());
      m_comp_delta[iConst] = UFOleadings.at(1)->p4().DeltaR(JconstWrapped.at(iConst)->rawConstituent()->p4());
      m_comp_pt[iConst] = log(JconstWrapped.at(iConst)->rawConstituent()->pt());
      m_comp_E[iConst] = log(JconstWrapped.at(iConst)->rawConstituent()->e());
      m_comp_pt_frac[iConst] = log(JconstWrapped.at(iConst)->rawConstituent()->pt()/UFOleadings.at(1)->pt());
      m_comp_E_frac[iConst] = log(JconstWrapped.at(iConst)->rawConstituent()->e()/UFOleadings.at(1)->e());
      const xAOD::TrackCaloCluster* ufo = dynamic_cast<const xAOD::TrackCaloCluster*>(JconstWrapped.at(iConst)->rawConstituent());
      m_comp_taste[iConst] = ufo->taste();
    }
    //___________HERE WE DEFINE OUR LABELS ! NEED TO BE EXTRA CAREFUL__________________
    m_truth_isW          = (UFOtruth_particles.at(1)->isW() && UFOtruth_matches.at(1)->m()/GeV > 50 && UFOtruth_matches.at(1)->m()/GeV < 100);
    m_truth_isZ          = (UFOtruth_particles.at(1)->isZ() && UFOtruth_matches.at(1)->m()/GeV > 60 && UFOtruth_matches.at(1)->m()/GeV < 100);
    m_truth_isV          = m_truth_isW || m_truth_isZ;
    m_truth_isQCD        = !m_truth_isV;
    if(m_truth_isV){
      m_truth_isLong       = m_isSigLong;
      m_truth_isTrans      = m_isSigTrans;
    }
    m_truth_isW_Long     = m_truth_isW && m_truth_isLong; 
    m_truth_isZ_Long     = m_truth_isZ && m_truth_isLong; 
    m_truth_isW_Trans    = m_truth_isW && m_truth_isTrans;
    m_truth_isZ_Trans    = m_truth_isZ && m_truth_isTrans;

    m_isLead                       = false; 
    writeTree("UFO");
  }   
  else 
    std::cout << "Problematic sublead event" << std::endl;

  return StatusCode::SUCCESS;
}

StatusCode CompTree::finalize() {
  ATH_MSG_INFO ("Finalizing " << name() << "...");

  return StatusCode::SUCCESS;
}

StatusCode CompTree::bookTree(std::string collectionName)
{
  ATH_MSG_DEBUG( "bookTree()" );
  
  // creating the tree for the jet collection
  TTree * tree = new TTree (collectionName.c_str(), "ntuple");
    
  // add the branches
  tree->Branch("EventNumber"          , &m_eventNumber           );    
  tree->Branch("EventWeight"          , &m_eventWeight           );    
  tree->Branch("eta"                  , &m_eta                   );
  tree->Branch("phi"                  , &m_phi                   );
  tree->Branch("E"                    , &m_E                     );
  tree->Branch("m"                    , &m_m                     );
  tree->Branch("pt"                   , &m_pt                    );
  tree->Branch("truth_isW_Long"       , &m_truth_isW_Long        );
  tree->Branch("truth_isZ_Long"       , &m_truth_isZ_Long        );   
  tree->Branch("truth_isW_Trans"      , &m_truth_isW_Trans       );
  tree->Branch("truth_isZ_Trans"      , &m_truth_isZ_Trans       );   
  tree->Branch("truth_isQCD"          , &m_truth_isQCD           );   
  tree->Branch("truth_isW"            , &m_truth_isW             );   
  tree->Branch("truth_isZ"            , &m_truth_isZ             );   
  tree->Branch("truth_isLong"         , &m_truth_isLong          );   
  tree->Branch("truth_isTrans"        , &m_truth_isTrans         );   
  tree->Branch("truth_isV"            , &m_truth_isV             );   
  tree->Branch("isLead"               , &m_isLead                );  
  for(int i = 0; i < m_nComp; ++i){
    TString name_br = "";
    name_br.Form("%d", i);
    tree->Branch("comp_eta_"+name_br     , &m_comp_eta[i]              );  
    tree->Branch("comp_phi_"+name_br     , &m_comp_phi[i]              );  
    tree->Branch("comp_delta_"+name_br   , &m_comp_delta[i]            );  
    tree->Branch("comp_pt_"+name_br      , &m_comp_pt[i]               );  
    tree->Branch("comp_E_"+name_br       , &m_comp_E[i]                );  
    tree->Branch("comp_pt_frac_"+name_br , &m_comp_pt_frac[i]          );  
    tree->Branch("comp_E_frac_"+name_br  , &m_comp_E_frac[i]           );  
    tree->Branch("comp_taste_"+name_br   , &m_comp_taste[i]            );  
  }
  m_trees.insert(std::pair<std::string, TTree*>(collectionName, tree));
  
  if (m_tHistSvc) {
    if((m_tHistSvc->regTree(m_stream+collectionName, m_trees.at(collectionName))).isFailure()) {
      ATH_MSG_ERROR( "initialize() Could not register the validation Tree!" );
      return StatusCode::FAILURE;
    }
  }
    
  ATH_MSG_INFO("Ntuple Tree booked and registered successfully for collection " << collectionName << "!");
  
  cleanVariables();
  
  return StatusCode::SUCCESS;
  
}

/* write the TTree */
void CompTree::writeTree(std::string collectionName) { 
  m_trees.at(collectionName)->Fill();  
  cleanVariables();  
}

/* clean the variables */
void CompTree::cleanVariables() {  
  
  m_Weight                       = -999999;
  m_theta_Z_0                    = -999999;
  m_theta_Zjet_0                 = -999999;
  m_theta_Z_1                    = -999999;
  m_theta_Zjet_1                 = -999999;
  m_theta_W_0                    = -999999;
  m_theta_Wjet_0                 = -999999;
  m_theta_W_1                    = -999999;
  m_theta_Wjet_1                 = -999999;
  m_eventNumber                  = -999999;
  m_eventWeight                  = -999999;
  m_eta                          = -999999;
  m_phi                          = -999999;
  m_E                            = -999999;
  m_m                            = -999999;
  m_pt                           = -999999;
  m_truth_isW_Long               = 0;
  m_truth_isZ_Long               = 0;
  m_truth_isW_Trans              = 0;
  m_truth_isZ_Trans              = 0;
  m_truth_isQCD                  = 0;
  m_truth_isW                    = 0;
  m_truth_isZ                    = 0;
  m_truth_isTrans                = 0;
  m_truth_isLong                 = 0;
  m_truth_isV                    = 0;
  m_isLead                       = 0;
  for(int i=0; i<m_nComp; ++i){
    m_comp_eta[i]         = -999;
    m_comp_phi[i]         = -999;
    m_comp_delta[i]       = -999;
    m_comp_pt[i]          = -999;
    m_comp_E[i]           = -999;
    m_comp_pt_frac[i]     = -999;
    m_comp_E_frac[i]      = -999;
    m_comp_taste[i]       = -999;
  }
}


template <class S, class T>
const T* CompTree::ClusterMatched(const xAOD::Jet* jet, const S* truths) {
  double minDeltaR = m_maxJetDR;
  const T* matched = nullptr;
  for (const auto& tomatch : *truths) {
    if (jet->p4().DeltaR(tomatch->p4()) < minDeltaR) {
      minDeltaR = jet->p4().DeltaR(tomatch->p4());
      matched = tomatch;
    }
  }
  return matched;
}

/**Calibrate and record a shallow copy of a given jet container */
const xAOD::JetContainer* CompTree::calibrateAndRecordShallowCopyJetCollection(const xAOD::JetContainer * jetContainer, const std::string name) {
  // create a shallow copy of the jet container
  std::pair< xAOD::JetContainer*, xAOD::ShallowAuxContainer* >  shallowCopy = xAOD::shallowCopyContainer(*jetContainer);
  xAOD::JetContainer *jetContainerShallowCopy           = shallowCopy.first;
  xAOD::ShallowAuxContainer *jetAuxContainerShallowCopy = shallowCopy.second;

  if( evtStore()->record(jetContainerShallowCopy, name+"_Calib").isFailure() ){
    ATH_MSG_WARNING("Unable to record JetCalibratedContainer: " << name+"_Calib");
    return 0;
  }
  if( evtStore()->record(jetAuxContainerShallowCopy, name+"_Calib"+"Aux.").isFailure() ){
    ATH_MSG_WARNING("Unable to record JetCalibratedAuxContainer: " << name+"_Calib"+"Aux.");
    return 0;
  }

  static SG::AuxElement::Accessor< xAOD::IParticleLink > accSetOriginLink ("originalObjectLink");
  static SG::AuxElement::Decorator< float > decJvt("JvtUpdate");

  for ( xAOD::Jet *shallowCopyJet : * jetContainerShallowCopy ) {
 
    if( m_UFOjetCalibration->applyCalibration(*shallowCopyJet).isFailure() ){
      ATH_MSG_WARNING( "Failed to apply calibration to the jet container");
      return 0;
    }
    const xAOD::IParticleLink originLink( *jetContainer, shallowCopyJet->index() );
    accSetOriginLink(*shallowCopyJet) = originLink;
  }

  if( evtStore()->setConst(jetContainerShallowCopy ).isFailure() ){
    ATH_MSG_WARNING( "Failed to set jetcalibCollection (" << name+"_Calib"+"Aux." << ")const in StoreGate!");
    return 0;
  }
  if( evtStore()->setConst(jetAuxContainerShallowCopy ).isFailure() ){
    ATH_MSG_WARNING( "Failed to set jetcalibCollection (" << name+"_Calib"+"Aux." << ")const in StoreGate!");
    return 0;
  }

  return jetContainerShallowCopy;
}





