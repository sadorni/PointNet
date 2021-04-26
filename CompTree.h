#ifndef MYPACKAGE_COMPTREE_H
#define MYPACKAGE_COMPTREE_H 

#include <AnaAlgorithm/AnaAlgorithm.h>
#include "xAODJet/JetContainer.h"
#include <vector>
#include "xAODCore/ShallowAuxContainer.h"
#include "xAODCore/ShallowCopy.h"
#include "AsgTools/AnaToolHandle.h"
#include "JetCalibTools/IJetCalibrationTool.h"

#include "GaudiKernel/ToolHandle.h"


// ROOT includes
#include "TF1.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TMath.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TRandom3.h"
#include "TString.h"
#include "AthLinks/ElementLink.h"
#include "xAODJet/Jet.h"
#include "xAODTracking/TrackParticle.h"
#include "xAODTruth/TruthParticle.h"

//fastjet includes
// for Soft Drop
#include "fastjet/contrib/SoftDrop.hh"
#include "fastjet/ClusterSequence.hh"

// for Trimming
#include <fastjet/tools/Filter.hh>

// for Pruning
#include <fastjet/tools/Pruner.hh>

// for ECF
#include <fastjet/contrib/EnergyCorrelator.hh>

// for NSubjettiness
#include <fastjet/contrib/Nsubjettiness.hh>

#include "fastjet/PseudoJet.hh"
#include "GaudiKernel/ITHistSvc.h"

const int   m_nComp = 100;


class IJetCalibrationTool;
class TTree;

class CompTree: public ::AthAlgorithm { 
  private:

  ITHistSvc*                            m_tHistSvc;
  std::string m_stream;



  /// calibration tool
  asg::AnaToolHandle<IJetCalibrationTool>  m_UFOjetCalibration;
  bool                                     m_applyCalibration;
  bool                                     m_applyTruthMatching;
  bool                                     m_isSigLong;
  bool                                     m_isSigTrans;
  std::string                              m_JZslice;
  //std::string                              m_eventsFile;
  //std::string                              m_eventsTree;
  //std::set<int>                            m_eventvector;

  float m_maxJetDR;
  float m_maxEta;
  float m_minPt;
  float m_minMass;
  float m_maxMass;

  template<class T> const T* getContainer( const std::string & containerName);
  /// Get the matched jet
  template <class S, class T> const T* ClusterMatched(const xAOD::Jet* jet, const S* truths);
  //Calibrate and record a shallow copy of a given jet container
  const xAOD::JetContainer* calibrateAndRecordShallowCopyJetCollection(const xAOD::JetContainer * jetContainer, const std::string name);

  std::map   < std::string, TTree* >    m_trees;
  
  int                                   m_eventNumber;
  float                                 m_theta_Z_0;
  float                                 m_theta_Zjet_0;
  float                                 m_theta_Z_1;
  float                                 m_theta_Zjet_1;
  float                                 m_theta_W_0;
  float                                 m_theta_Wjet_0;
  float                                 m_theta_W_1;
  float                                 m_theta_Wjet_1;
  float                                 m_Weight;
  float                                 m_eventWeight;
  float                                 m_eta;
  float                                 m_phi;
  float                                 m_E  ;
  float                                 m_m  ;
  float                                 m_pt ;
  bool                                  m_truth_isW_Long;
  bool                                  m_truth_isZ_Long;
  bool                                  m_truth_isW_Trans;
  bool                                  m_truth_isZ_Trans;
  bool                                  m_truth_isQCD;
  bool                                  m_truth_isW;
  bool                                  m_truth_isZ;
  bool                                  m_truth_isLong;
  bool                                  m_truth_isTrans;
  bool                                  m_truth_isV;
  bool                                  m_isLead; 
  double                                m_comp_eta[m_nComp];
  double                                m_comp_phi[m_nComp];
  double                                m_comp_delta[m_nComp];
  double                                m_comp_pt[m_nComp];
  double                                m_comp_E[m_nComp];
  double                                m_comp_pt_frac[m_nComp];
  double                                m_comp_E_frac[m_nComp];
  double                                m_comp_taste[m_nComp];
  
  public: 
  CompTree( const std::string& name, ISvcLocator* pSvcLocator );
  virtual ~CompTree(); 

  /** standard Athena-Algorithm method */
  StatusCode          initialize();

  /** standard Athena-Algorithm method */
  StatusCode          execute();
    
  /** standard Athena-Algorithm method */
  StatusCode          finalize();

  /* book the TTree branches */
  virtual StatusCode  bookTree(std::string collectionName);
  
  /* write the TTree */
  virtual void writeTree(std::string collectionName);

  /* write the TTree */
  virtual void cleanVariables();


}; 

#endif

template<class T>
inline const T* CompTree::getContainer(const std::string & containerName){
    const T * ptr = evtStore()->retrieve< const T >( containerName );
      if (!ptr) {
            ATH_MSG_WARNING("Container '"<<containerName<<"' could not be retrieved");
              }
        return ptr;
}

