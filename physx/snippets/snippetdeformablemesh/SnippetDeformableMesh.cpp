//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2019 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.  

// ****************************************************************************
// This snippet shows how to use deformable meshes in PhysX.
// ****************************************************************************

#include <ctype.h>

#include "PxPhysicsAPI.h"

#include "../snippetcommon/SnippetPrint.h"
#include "../snippetcommon/SnippetPVD.h"
#include "../snippetutils/SnippetUtils.h"


#include "vec.h"
#include "mesh.h"
#include "wavefront.h"
#include "orbiter.h"
//
#include "image.h"
#include "image_io.h"
#include "image_hdr.h"

using namespace physx;

PxDefaultAllocator		gAllocator;
PxDefaultErrorCallback	gErrorCallback;

PxFoundation*			gFoundation = NULL;
PxPhysics*				gPhysics	= NULL;
PxCooking*				gCooking	= NULL;

PxDefaultCpuDispatcher*	gDispatcher = NULL;
PxScene*				gScene		= NULL;

PxMaterial*				gMaterial	= NULL;

PxPvd*                  gPvd        = NULL;

PxTriangleMesh*			gMesh		= NULL;
PxRigidStatic*			gActor		= NULL;

PxReal stackZ = 10.0f;

static const PxU32 gGridSize = 2;
static const PxReal gGridStep = 512.0f / PxReal(gGridSize-1);
static float gTime = 0.0f;

static PxRigidDynamic* createDynamic(const PxTransform& t, const PxGeometry& geometry, const PxVec3& velocity=PxVec3(0), PxReal density=1.0f)
{
	PxRigidDynamic* dynamic = PxCreateDynamic(*gPhysics, t, geometry, *gMaterial, density);
	dynamic->setLinearVelocity(velocity);
	gScene->addActor(*dynamic);
	return dynamic;
}

static void createStack(const PxTransform& t, PxU32 size, PxReal halfExtent)
{
	PxShape* shape = gPhysics->createShape(PxBoxGeometry(halfExtent, halfExtent, halfExtent), *gMaterial);
	for(PxU32 i=0; i<size;i++)
	{
		for(PxU32 j=0;j<size-i;j++)
		{
			PxTransform localTm(PxVec3(PxReal(j*2) - PxReal(size-i), PxReal(i*2+1), 0) * halfExtent);
			PxRigidDynamic* body = gPhysics->createRigidDynamic(t.transform(localTm));
			body->attachShape(*shape);
			PxRigidBodyExt::updateMassAndInertia(*body, 10.0f);
			gScene->addActor(*body);
		}
	}
	shape->release();
}

struct Triangle
{
	PxU32 ind0, ind1, ind2;
};

static void updateVertices(PxVec3* verts, float amplitude=0.0f)
{
	const PxU32 gridSize = gGridSize;
	const PxReal gridStep = gGridStep;

	for(PxU32 a=0; a<gridSize; a++)
	{
		const float coeffA = float(a)/float(gridSize);
		for(PxU32 b=0; b<gridSize; b++)
		{
			const float coeffB = float(b)/float(gridSize);

			const float y = 20.0f + sinf(coeffA*PxTwoPi)*cosf(coeffB*PxTwoPi)*amplitude;

			verts[a * gridSize + b] = PxVec3(-400.0f + b*gridStep, y, -400.0f + a*gridStep);
		}
	}
}

// Setup common cooking params
void setupCommonCookingParams(PxCookingParams& params, bool skipMeshCleanup, bool skipEdgeData)
{
	// we suppress the triangle mesh remap table computation to gain some speed, as we will not need it 
// in this snippet
	params.suppressTriangleMeshRemapTable = true;

	// If DISABLE_CLEAN_MESH is set, the mesh is not cleaned during the cooking. The input mesh must be valid. 
	// The following conditions are true for a valid triangle mesh :
	//  1. There are no duplicate vertices(within specified vertexWeldTolerance.See PxCookingParams::meshWeldTolerance)
	//  2. There are no large triangles(within specified PxTolerancesScale.)
	// It is recommended to run a separate validation check in debug/checked builds, see below.

	if (!skipMeshCleanup)
		params.meshPreprocessParams &= ~static_cast<PxMeshPreprocessingFlags>(PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH);
	else
		params.meshPreprocessParams |= PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH;

	// If DISABLE_ACTIVE_EDGES_PREDOCOMPUTE is set, the cooking does not compute the active (convex) edges, and instead 
	// marks all edges as active. This makes cooking faster but can slow down contact generation. This flag may change 
	// the collision behavior, as all edges of the triangle mesh will now be considered active.
	if (!skipEdgeData)
		params.meshPreprocessParams &= ~static_cast<PxMeshPreprocessingFlags>(PxMeshPreprocessingFlag::eDISABLE_ACTIVE_EDGES_PRECOMPUTE);
	else
		params.meshPreprocessParams |= PxMeshPreprocessingFlag::eDISABLE_ACTIVE_EDGES_PRECOMPUTE;
}


// Creates a triangle mesh using BVH33 midphase with different settings.
static PxTriangleMesh* createBV33TriangleMesh(PxU32 numVertices, const PxVec3* vertices, PxU32 numTriangles, const PxU32* indices,
	bool skipMeshCleanup, bool skipEdgeData, bool inserted, bool cookingPerformance, bool meshSizePerfTradeoff)
{
	PxU64 startTime = SnippetUtils::getCurrentTimeCounterValue();

	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = numVertices;
	meshDesc.points.data = vertices;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.triangles.count = numTriangles;
	meshDesc.triangles.data = indices;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);

	PxCookingParams params = gCooking->getParams();

	// Create BVH33 midphase
	params.midphaseDesc = PxMeshMidPhase::eBVH33;

	// setup common cooking params
	setupCommonCookingParams(params, skipMeshCleanup, skipEdgeData);

	// The COOKING_PERFORMANCE flag for BVH33 midphase enables a fast cooking path at the expense of somewhat lower quality BVH construction.	
	if (cookingPerformance)
		params.midphaseDesc.mBVH33Desc.meshCookingHint = PxMeshCookingHint::eCOOKING_PERFORMANCE;
	else
		params.midphaseDesc.mBVH33Desc.meshCookingHint = PxMeshCookingHint::eSIM_PERFORMANCE;

	// If meshSizePerfTradeoff is set to true, smaller mesh cooked mesh is produced. The mesh size/performance trade-off
	// is controlled by setting the meshSizePerformanceTradeOff from 0.0f (smaller mesh) to 1.0f (larger mesh).
	if (meshSizePerfTradeoff)
	{
		params.midphaseDesc.mBVH33Desc.meshSizePerformanceTradeOff = 0.0f;
	}
	else
	{
		// using the default value
		params.midphaseDesc.mBVH33Desc.meshSizePerformanceTradeOff = 0.55f;
	}

	gCooking->setParams(params);

	PX_ASSERT(gCooking->validateTriangleMesh(meshDesc));


	PxTriangleMesh* triMesh = NULL;
	PxU32 meshSize = 0;

	// The cooked mesh may either be saved to a stream for later loading, or inserted directly into PxPhysics.
	if (inserted)
	{
		triMesh = gCooking->createTriangleMesh(meshDesc, gPhysics->getPhysicsInsertionCallback());
	}
	else
	{
		PxDefaultMemoryOutputStream outBuffer;
		gCooking->cookTriangleMesh(meshDesc, outBuffer);

		PxDefaultMemoryInputData stream(outBuffer.getData(), outBuffer.getSize());
		triMesh = gPhysics->createTriangleMesh(stream);

		meshSize = outBuffer.getSize();
	}


	// Print the elapsed time for comparison
	PxU64 stopTime = SnippetUtils::getCurrentTimeCounterValue();
	float elapsedTime = SnippetUtils::getElapsedTimeInMilliseconds(stopTime - startTime);
	printf("\t -----------------------------------------------\n");
	printf("\t Create triangle mesh with %d triangles: \n", numTriangles);
	cookingPerformance ? printf("\t\t Cooking performance on\n") : printf("\t\t Cooking performance off\n");
	inserted ? printf("\t\t Mesh inserted on\n") : printf("\t\t Mesh inserted off\n");
	!skipEdgeData ? printf("\t\t Precompute edge data on\n") : printf("\t\t Precompute edge data off\n");
	!skipMeshCleanup ? printf("\t\t Mesh cleanup on\n") : printf("\t\t Mesh cleanup off\n");
	printf("\t\t Mesh size/performance trade-off: %f \n", double(params.midphaseDesc.mBVH33Desc.meshSizePerformanceTradeOff));
	printf("\t Elapsed time in ms: %f \n", double(elapsedTime));
	if (!inserted)
	{
		printf("\t Mesh size: %d \n", meshSize);
	}

	return(triMesh);
	//triMesh->release();
}


// Creates a triangle mesh using BVH34 midphase with different settings.
static PxTriangleMesh* createBV34TriangleMesh(PxU32 numVertices, const PxVec3* vertices, PxU32 numTriangles, const PxU32* indices,
	bool skipMeshCleanup, bool skipEdgeData, bool inserted, const PxU32 numTrisPerLeaf)
{
	PxU64 startTime = SnippetUtils::getCurrentTimeCounterValue();

	PxTriangleMeshDesc meshDesc;
	meshDesc.points.count = numVertices;
	meshDesc.points.data = vertices;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.triangles.count = numTriangles;
	meshDesc.triangles.data = indices;
	meshDesc.triangles.stride = 3 * sizeof(PxU32);

	PxCookingParams params = gCooking->getParams();

	// Create BVH34 midphase
	params.midphaseDesc = PxMeshMidPhase::eBVH34;

	// setup common cooking params
	setupCommonCookingParams(params, skipMeshCleanup, skipEdgeData);

	// Cooking mesh with less triangles per leaf produces larger meshes with better runtime performance
	// and worse cooking performance. Cooking time is better when more triangles per leaf are used.
	params.midphaseDesc.mBVH34Desc.numPrimsPerLeaf = numTrisPerLeaf;

	gCooking->setParams(params);

	PX_ASSERT(gCooking->validateTriangleMesh(meshDesc));

	PxTriangleMesh* triMesh = NULL;
	PxU32 meshSize = 0;

	// The cooked mesh may either be saved to a stream for later loading, or inserted directly into PxPhysics.
	//if (inserted)
	//{
	//triMesh = gCooking->createTriangleMesh(meshDesc, gPhysics->getPhysicsInsertionCallback());
	//}
	//else
	//{
	PxDefaultMemoryOutputStream outBuffer;
	gCooking->cookTriangleMesh(meshDesc, outBuffer);

	PxDefaultMemoryInputData stream(outBuffer.getData(), outBuffer.getSize());
	triMesh = gPhysics->createTriangleMesh(stream);

	meshSize = outBuffer.getSize();
	//}

	// Print the elapsed time for comparison
	PxU64 stopTime = SnippetUtils::getCurrentTimeCounterValue();
	float elapsedTime = SnippetUtils::getElapsedTimeInMilliseconds(stopTime - startTime);
	printf("\t -----------------------------------------------\n");
	printf("\t Create triangle mesh with %d triangles: \n", numTriangles);
	inserted ? printf("\t\t Mesh inserted on\n") : printf("\t\t Mesh inserted off\n");
	!skipEdgeData ? printf("\t\t Precompute edge data on\n") : printf("\t\t Precompute edge data off\n");
	!skipMeshCleanup ? printf("\t\t Mesh cleanup on\n") : printf("\t\t Mesh cleanup off\n");
	printf("\t\t Num triangles per leaf: %d \n", numTrisPerLeaf);
	printf("\t Elapsed time in ms: %f \n", double(elapsedTime));
	if (!inserted)
	{
		printf("\t Mesh size: %d \n", meshSize);
	}

	return(triMesh);

	//triMesh->release();
}


static PxTriangleMesh* createMeshGround()
{
	const PxU32 gridSize = gGridSize;

	PxVec3 verts[gridSize * gridSize];

	const PxU32 nbTriangles = 2 * (gridSize - 1) * (gridSize-1);

	Triangle indices[nbTriangles];

	updateVertices(verts);

	for (PxU32 a = 0; a < (gridSize-1); ++a)
	{
		for (PxU32 b = 0; b < (gridSize-1); ++b)
		{


			Triangle& tri0 = indices[(a * (gridSize-1) + b) * 2];
			Triangle& tri1 = indices[((a * (gridSize-1) + b) * 2) + 1];

			tri0.ind0 = a * gridSize + b + 1;
			tri0.ind1 = a * gridSize + b;
			tri0.ind2 = (a + 1) * gridSize + b + 1;

			tri1.ind0 = (a + 1) * gridSize + b + 1;
			tri1.ind1 = a * gridSize + b;
			tri1.ind2 = (a + 1) * gridSize + b;

		}
	}

	for (PxU32 i = 0; i < nbTriangles; i++)
	{
		//std::cout << meshOBJ.triangle(i).a.x << " " << meshOBJ.triangle(i).a.y << " " << meshOBJ.triangle(i).a.z << std::endl;
		//std::cout << va.x << " " << va.y << " " << va.z << std::endl;
		std::cout << "true indice tri 1 : " << indices[i].ind0 << std::endl;
		std::cout << "true indice tri 2 : " << indices[i].ind1 << std::endl;
		std::cout << "true indice tri 3 : " << indices[i].ind2 << std::endl;
	}

	for (PxU32 i = 0; i < gridSize * gridSize; i++)
	{
		//std::cout << meshOBJ.triangle(i).a.x << " " << meshOBJ.triangle(i).a.y << " " << meshOBJ.triangle(i).a.z << std::endl;
		//std::cout << va.x << " " << va.y << " " << va.z << std::endl;
		std::cout << "true vert : " << verts[i].x << " " << verts[i].y << " " << verts[i].z << std::endl;

	}

	PxTriangleMeshDesc meshDesc;
	meshDesc.points.data = verts;
	meshDesc.points.count = gridSize * gridSize;
	meshDesc.points.stride = sizeof(PxVec3);
	meshDesc.triangles.count = nbTriangles;
	meshDesc.triangles.data = indices;
	meshDesc.triangles.stride = sizeof(Triangle);


	PX_ASSERT(gCooking->validateTriangleMesh(meshDesc));


	PxTriangleMesh* triMesh = gCooking->createTriangleMesh(meshDesc, gPhysics->getPhysicsInsertionCallback());

	return triMesh;
}

void initPhysics(bool /*interactive*/)
{
	gFoundation = PxCreateFoundation(PX_PHYSICS_VERSION, gAllocator, gErrorCallback);

	gPvd = PxCreatePvd(*gFoundation);
	PxPvdTransport* transport = PxDefaultPvdSocketTransportCreate(PVD_HOST, 5425, 10);
	gPvd->connect(*transport,PxPvdInstrumentationFlag::eALL);

	gPhysics = PxCreatePhysics(PX_PHYSICS_VERSION, *gFoundation, PxTolerancesScale(), true, gPvd);

	PxCookingParams cookingParams(gPhysics->getTolerancesScale());

	// Deformable meshes are only supported with PxMeshMidPhase::eBVH33.
	cookingParams.midphaseDesc.setToDefault(PxMeshMidPhase::eBVH33);
	// We need to disable the mesh cleaning part so that the vertex mapping remains untouched.
	cookingParams.meshPreprocessParams	= PxMeshPreprocessingFlag::eDISABLE_CLEAN_MESH;

	gCooking = PxCreateCooking(PX_PHYSICS_VERSION, *gFoundation, cookingParams);

	PxSceneDesc sceneDesc(gPhysics->getTolerancesScale());
	sceneDesc.gravity = PxVec3(0.0f, -9.81f, 0.0f);
	gDispatcher = PxDefaultCpuDispatcherCreate(2);
	sceneDesc.cpuDispatcher	= gDispatcher;
	sceneDesc.filterShader	= PxDefaultSimulationFilterShader;

	gScene = gPhysics->createScene(sceneDesc);

	PxPvdSceneClient* pvdClient = gScene->getScenePvdClient();
	if(pvdClient)
	{
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONSTRAINTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_CONTACTS, true);
		pvdClient->setScenePvdFlag(PxPvdSceneFlag::eTRANSMIT_SCENEQUERIES, true);
	}

	gMaterial = gPhysics->createMaterial(0.5f, 0.5f, 0.6f);


	const char* mesh_filename = "C:\\Users\\PC-B\\Documents\\Guillaume_ITB\\Synthese-Image\\data\\triangle.obj";
	Mesh meshOBJ = read_mesh(mesh_filename);
	//if (mesh.triangle_count() == 0) {
		// erreur de chargement, pas de triangles
	const PxU32 numVerticesOBJ = PxU32(meshOBJ.vertex_count());
	const PxU32 numTrianglesOBJ = PxU32(meshOBJ.triangle_count());
	std::cout << "numTrianglesOBJ " << numTrianglesOBJ << std::endl;
	std::cout << "numVerticesOBJ " << numVerticesOBJ << std::endl;

	PxVec3* verticesOBJ = new PxVec3[numVerticesOBJ];
	PxU32* indicesOBJ = new PxU32[numVerticesOBJ];

	//PxU32 currentIdx = 0;
	//for (int i = 0; i <= numVerticesOBJ; i++)
	//{
	//	const PxVec3 v = PxVec3(mesh.positions()[i].x, mesh.positions()[i].y, mesh.positions()[i].z);
	//	verticesOBJ[i] = v;
	//}

	for (int i = 0; i < numTrianglesOBJ; i++)
	{


		indicesOBJ[3 * i] = PxU32(3 * i);
		const PxVec3 va = PxVec3(meshOBJ.triangle(i).a.x, meshOBJ.triangle(i).a.y, meshOBJ.triangle(i).a.z);
		std::cout << meshOBJ.triangle(i).a.x << " " << meshOBJ.triangle(i).a.y << " " << meshOBJ.triangle(i).a.z << std::endl;
		std::cout << va.x << " " << va.y << " " << va.z << std::endl;
		std::cout << "indice tri 1 : " << indicesOBJ[3 * i] << std::endl;
		verticesOBJ[3 * i] = va;
		indicesOBJ[3 * i + 1] = PxU32(3 * i + 1);
		const PxVec3 vb = PxVec3(meshOBJ.triangle(i).b.x, meshOBJ.triangle(i).b.y, meshOBJ.triangle(i).b.z);
		std::cout << meshOBJ.triangle(i).b.x << " " << meshOBJ.triangle(i).b.y << " " << meshOBJ.triangle(i).b.z << std::endl;
		std::cout << vb.x << " " << vb.y << " " << vb.z << std::endl;
		std::cout << "indice tri 2 : " << indicesOBJ[3 * i + 1] << std::endl;
		verticesOBJ[3 * i + 1] = vb;
		indicesOBJ[3 * i + 2] = PxU32(3 * i + 2);
		const PxVec3 vc = PxVec3(meshOBJ.triangle(i).c.x, meshOBJ.triangle(i).c.y, meshOBJ.triangle(i).c.z);
		std::cout << meshOBJ.triangle(i).c.x << " " << meshOBJ.triangle(i).c.y << " " << meshOBJ.triangle(i).c.z << std::endl;
		std::cout << vc.x << " " << vc.y << " " << vc.z << std::endl;
		std::cout << "indice tri 3 : " << indicesOBJ[3 * i + 2] << std::endl;
		verticesOBJ[3 * i + 2] = vc;
	}


	//const PxVec3 v1 = PxVec3(-400, 20, -400);
	//const PxVec3 v2 = PxVec3(112, 20, -400);
	//const PxVec3 v3 = PxVec3(-400, 20, 112);
	//const PxVec3 v4 = PxVec3(112, 20, 112);


	//verticesOBJ[PxU32(0)] = v1;
	//verticesOBJ[PxU32(1)] = v2;
	//verticesOBJ[PxU32(2)] = v3;
	//verticesOBJ[PxU32(3)] = v4;

	//indicesOBJ[PxU32(0)] = PxU32(1);
	//indicesOBJ[PxU32(1)] = PxU32(0);
	//indicesOBJ[PxU32(2)] = PxU32(3);
	//indicesOBJ[PxU32(3)] = PxU32(3);
	//indicesOBJ[PxU32(4)] = PxU32(0);
	//indicesOBJ[PxU32(5)] = PxU32(2);


	//PxTriangleMesh* mesh = createMeshGround();

	PxTriangleMesh* mesh = createBV33TriangleMesh(numVerticesOBJ, verticesOBJ, numTrianglesOBJ, indicesOBJ, false, false, false, false, false);

	gMesh = mesh;

	PxTriangleMeshGeometry geom(mesh);

	PxRigidStatic* groundMesh = gPhysics->createRigidStatic(PxTransform(PxVec3(0, 2, 0)));
	gActor = groundMesh;
	PxShape* shape = gPhysics->createShape(geom, *gMaterial);

	{
		shape->setContactOffset(0.02f);
		// A negative rest offset helps to avoid jittering when the deformed mesh moves away from objects resting on it.
		shape->setRestOffset(-0.5f);
	}

	groundMesh->attachShape(*shape);
	gScene->addActor(*groundMesh);

	createStack(PxTransform(PxVec3(0,22,0)), 10, 2.0f);
}

void stepPhysics(bool /*interactive*/)
{
	{
		PxVec3* verts = gMesh->getVerticesForModification();
		gTime += 0.01f;
		updateVertices(verts, sinf(gTime)*20.0f);
		PxBounds3 newBounds = gMesh->refitBVH();
		PX_UNUSED(newBounds);

		// Reset filtering to tell the broadphase about the new mesh bounds.
		gScene->resetFiltering(*gActor);
	}
	gScene->simulate(1.0f/60.0f);
	gScene->fetchResults(true);
}
	
void cleanupPhysics(bool /*interactive*/)
{
	PX_RELEASE(gScene);
	PX_RELEASE(gDispatcher);
	PX_RELEASE(gPhysics);
	PX_RELEASE(gCooking);
	if(gPvd)
	{
		PxPvdTransport* transport = gPvd->getTransport();
		gPvd->release();	gPvd = NULL;
		PX_RELEASE(transport);
	}
	PX_RELEASE(gFoundation);
	
	printf("SnippetDeformableMesh done.\n");
}

void keyPress(unsigned char key, const PxTransform& camera)
{
	switch(toupper(key))
	{
	case ' ':	createDynamic(camera, PxSphereGeometry(3.0f), camera.rotate(PxVec3(0,0,-1))*200, 3.0f);	break;
	}
}

int snippetMain(int, const char*const*)
{
#ifdef RENDER_SNIPPET
	extern void renderLoop();
	renderLoop();
#else
	static const PxU32 frameCount = 100;
	initPhysics(false);
	for(PxU32 i=0; i<frameCount; i++)
		stepPhysics(false);
	cleanupPhysics(false);
#endif

	return 0;
}
