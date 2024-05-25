#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include <map>
using namespace mlir;

namespace {
class PolozovMaxDepthPass
    : public PassWrapper<PolozovMaxDepthPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { return "PolozovMaxDepthPass"; }
  StringRef getDescription() const final {
    return "Counts the max depth of region nests in the function.";
  }
  std::map<Operation *, int>Depth;
  void runOnOperation() override {
    
    Operation *func = getOperation();

    int maxDepth = 1;
    func->walk([&](Operation *cur) {
      maxDepth = std::max(maxDepth, getMaxDepth(cur));
      //llvm::outs()<<(*op)<<"\n\n\n";
      //cmake -S ../llvm-nnsu-2024/llvm -B . -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS='llvm' -DLLVM_TARGETS_TO_BUILD='X86'
    });
    func->setAttr(
      "maxDepth",
      IntegerAttr::get(IntegerType::get(func->getContext(), 32), maxDepth));
  }
  private:
  int getMaxDepth(Operation *op){
    if(Depth.count(op)) {
      return Depth[op];
    }
    int& answer = Depth[op];
    answer = 1;
    if(op->getParentOp()){
      answer = std::max(answer, 1 + getMaxDepth(op->getParentOp()));
    }
    return answer;
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(PolozovMaxDepthPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(PolozovMaxDepthPass)

PassPluginLibraryInfo getPolozovMaxDepthPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "PolozovMaxDepthPass", LLVM_VERSION_STRING,
          []() { PassRegistration<PolozovMaxDepthPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getPolozovMaxDepthPassPluginInfo();
}