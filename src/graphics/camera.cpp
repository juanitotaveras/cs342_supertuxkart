//
//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2004-2015 Steve Baker <sjbaker1@airmail.net>
//  Copyright (C) 2006-2015 SuperTuxKart-Team, Steve Baker
//
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public License
//  as published by the Free Software Foundation; either version 3
//  of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

#include "graphics/camera.hpp"

#include "config/stk_config.hpp"
#include "config/user_config.hpp"
#include "graphics/camera_debug.hpp"
#include "graphics/camera_end.hpp"
#include "graphics/camera_fps.hpp"
#include "graphics/camera_normal.hpp"
#include "graphics/irr_driver.hpp"
#include "io/xml_node.hpp"
#include "karts/abstract_kart.hpp"
#include "karts/explosion_animation.hpp"
#include "karts/kart.hpp"
#include "karts/kart_properties.hpp"
#include "karts/skidding.hpp"
#include "physics/btKart.hpp"
#include "race/race_manager.hpp"
#include "tracks/track.hpp"
#include "utils/aligned_array.hpp"
#include "utils/constants.hpp"
#include "utils/vs.hpp"

#include "ISceneManager.h"

#include <cmath>

std::vector<Camera*> Camera::m_all_cameras;
Camera*              Camera::s_active_camera = NULL;
Camera::CameraType   Camera::m_default_type  = Camera::CM_TYPE_NORMAL;

// ------------------------------------------------------------------------
/** Creates a new camera and adds it to the list of all cameras. Also the
 *  camera index (which determines which viewport to use in split screen)
 *  is set.
 */
Camera* Camera::createCamera(AbstractKart* kart)
{
    Camera *camera = createCamera((int)m_all_cameras.size(), 
                                   m_default_type, kart      );
    m_all_cameras.push_back(camera);
    return camera;
}   // createCamera(kart)

// ----------------------------------------------------------------------------
/** Creates a camera of the specified type, but does not add it to the list
 *  of all cameras. This is a helper function for other static functions.
 *  \paran index Index this camera has in the list of all cameras.
 *  \param type The camera type of the camera to create.
 *  \param kart To which kart the camera is attached (NULL if a free camera).
 */
Camera* Camera::createCamera(unsigned int index, CameraType type,
                             AbstractKart* kart)
{
    Camera *camera = NULL;
    switch (type)
    {
    case CM_TYPE_NORMAL: camera = new CameraNormal(CM_TYPE_NORMAL, index, kart);
                                                                 break;
    case CM_TYPE_DEBUG:  camera = new CameraDebug (index, kart); break;
    case CM_TYPE_FPS:    camera = new CameraFPS   (index, kart); break;
    case CM_TYPE_END:    camera = new CameraEnd   (index, kart); break;
    }   // switch type

    return camera;
}   // createCamera

// ----------------------------------------------------------------------------
void Camera::changeCamera(unsigned int camera_index, CameraType type)
{
    assert(camera_index<m_all_cameras.size());

    Camera *old_camera = m_all_cameras[camera_index];
    // Nothing to do if this is already the right type.

    if(old_camera->getType()==type) return;

    Camera *new_camera = createCamera(old_camera->getIndex(), type,
                                      old_camera->m_original_kart);
    // Replace the previous camera
    m_all_cameras[camera_index] = new_camera;
    if(s_active_camera == old_camera)
        s_active_camera = new_camera;
    delete old_camera;
}   // changeCamera

// ----------------------------------------------------------------------------
void Camera::resetAllCameras()
{
    for (unsigned int i = 0; i < Camera::getNumCameras(); i++)
    {
        changeCamera(i, m_default_type);
        getCamera(i)->reset();
    }
}   // resetAllCameras

// ----------------------------------------------------------------------------
Camera::Camera(CameraType type, int camera_index, AbstractKart* kart) 
      : m_kart(NULL)
{
    m_mode          = CM_NORMAL;
    m_type          = type;
    m_index         = camera_index;
    m_original_kart = kart;
    m_camera        = irr_driver->addCameraSceneNode();
    m_previous_pv_matrix = core::matrix4();

    setupCamera();
    setKart(kart);
    m_ambient_light = Track::getCurrentTrack()->getDefaultAmbientColor();

    reset();
}   // Camera

// ----------------------------------------------------------------------------
/** Removes the camera scene node from the scene.
 */
Camera::~Camera()
{
    irr_driver->removeCameraSceneNode(m_camera);

    if (s_active_camera == this)
        s_active_camera = NULL;
}   // ~Camera

//-----------------------------------------------------------------------------
/** Changes the owner of this camera to the new kart.
 *  \param new_kart The new kart to use this camera.
 */
void Camera::setKart(AbstractKart *new_kart)
{
    m_kart = new_kart;
#ifdef DEBUG
    std::string name = new_kart ? new_kart->getIdent()+"'s camera"
                                : "Unattached camera";
    getCameraSceneNode()->setName(name.c_str());
#endif

}   // setKart

//-----------------------------------------------------------------------------
/** Sets up the viewport, aspect ratio, field of view, and scaling for this
 *  camera.
 */
void Camera::setupCamera()
{
    m_aspect = (float)(irr_driver->getActualScreenSize().Width)
             /         irr_driver->getActualScreenSize().Height;
    switch(race_manager->getNumLocalPlayers())
    {
    case 1: m_viewport = core::recti(0, 0,
                                     irr_driver->getActualScreenSize().Width,
                                     irr_driver->getActualScreenSize().Height);
            m_scaling  = core::vector2df(1.0f, 1.0f);
            m_fov      = DEGREE_TO_RAD*stk_config->m_camera_fov[0];
            break;
    case 2: m_viewport = core::recti(0,
                                     m_index==0 ? 0
                                                : irr_driver->getActualScreenSize().Height>>1,
                                     irr_driver->getActualScreenSize().Width,
                                     m_index==0 ? irr_driver->getActualScreenSize().Height>>1
                                                : irr_driver->getActualScreenSize().Height);
            m_scaling  = core::vector2df(1.0f, 0.5f);
            m_aspect  *= 2.0f;
            m_fov      = DEGREE_TO_RAD*stk_config->m_camera_fov[1];
            break;
    case 3:
            /*
            if(m_index<2)
            {
                m_viewport = core::recti(m_index==0 ? 0
                                                    : irr_driver->getActualScreenSize().Width>>1,
                                         0,
                                         m_index==0 ? irr_driver->getActualScreenSize().Width>>1
                                                    : irr_driver->getActualScreenSize().Width,
                                         irr_driver->getActualScreenSize().Height>>1);
                m_scaling  = core::vector2df(0.5f, 0.5f);
                m_fov      = DEGREE_TO_RAD*50.0f;
            }
            else
            {
                m_viewport = core::recti(0, irr_driver->getActualScreenSize().Height>>1,
                                         irr_driver->getActualScreenSize().Width,
                                         irr_driver->getActualScreenSize().Height);
                m_scaling  = core::vector2df(1.0f, 0.5f);
                m_fov      = DEGREE_TO_RAD*65.0f;
                m_aspect  *= 2.0f;
            }
            break;*/
    case 4:
            { // g++ 4.3 whines about the variables in switch/case if not {}-wrapped (???)
            const int x1 = (m_index%2==0 ? 0 : irr_driver->getActualScreenSize().Width>>1);
            const int y1 = (m_index<2    ? 0 : irr_driver->getActualScreenSize().Height>>1);
            const int x2 = (m_index%2==0 ? irr_driver->getActualScreenSize().Width>>1  : irr_driver->getActualScreenSize().Width);
            const int y2 = (m_index<2    ? irr_driver->getActualScreenSize().Height>>1 : irr_driver->getActualScreenSize().Height);
            m_viewport = core::recti(x1, y1, x2, y2);
            m_scaling  = core::vector2df(0.5f, 0.5f);
            m_fov      = DEGREE_TO_RAD*stk_config->m_camera_fov[3];
            }
            break;
    default:
            if(UserConfigParams::logMisc())
                Log::warn("Camera", "Incorrect number of players: '%d' - assuming 1.",
                          race_manager->getNumLocalPlayers());
            m_viewport = core::recti(0, 0,
                                     irr_driver->getActualScreenSize().Width,
                                     irr_driver->getActualScreenSize().Height);
            m_scaling  = core::vector2df(1.0f, 1.0f);
            m_fov      = DEGREE_TO_RAD*75.0f;
            break;
    }   // switch
    m_camera->setFOV(m_fov);
    m_camera->setAspectRatio(m_aspect);
    m_camera->setFarValue(Track::getCurrentTrack()->getCameraFar());
}   // setupCamera

// ----------------------------------------------------------------------------
/** Sets the mode of the camera.
 *  \param mode Mode the camera should be switched to.
 */
void Camera::setMode(Mode mode)
{
    // If we switch from reverse view, move the camera immediately to the
    // correct position.
    if( (m_mode==CM_REVERSE && mode==CM_NORMAL) || 
        (m_mode==CM_FALLING && mode==CM_NORMAL)    )
    {
        Vec3 start_offset(0, 1.6f, -3);
        Vec3 current_position = m_kart->getTrans()(start_offset);
        Vec3 target_position = m_kart->getTrans()(Vec3(0, 0, 1));
        // Don't set position and target the same, otherwise
        // nan values will be calculated in ViewArea of camera
        m_camera->setPosition(current_position.toIrrVector());
        m_camera->setTarget(target_position.toIrrVector());
    }

    m_mode = mode;
}   // setMode

// ----------------------------------------------------------------------------
/** Returns the current mode of the camera.
 */
Camera::Mode Camera::getMode()
{
    return m_mode;
}   // getMode

//-----------------------------------------------------------------------------
/** Reset is called when a new race starts. Make sure that the camera
    is aligned neutral, and not like in the previous race
*/
void Camera::reset()
{
    m_kart = m_original_kart;
    setMode(CM_NORMAL);

    if (m_kart != NULL)
        setInitialTransform();
}   // reset

//-----------------------------------------------------------------------------
/** Saves the current kart position as initial starting position for the
 *  camera.
 */
void Camera::setInitialTransform()
{
    if (m_kart == NULL) return;
    Vec3 start_offset(0, 1.6f, -3);
    Vec3 current_position = m_kart->getTrans()(start_offset);
    assert(!std::isnan(current_position.getX()));
    assert(!std::isnan(current_position.getY()));
    assert(!std::isnan(current_position.getZ()));
    m_camera->setPosition(  current_position.toIrrVector());
    // Reset the target from the previous target (in case of a restart
    // of a race) - otherwise the camera will initially point in the wrong
    // direction till smoothMoveCamera has corrected this. Setting target
    // to position doesn't make sense, but smoothMoves will adjust the
    // value before the first frame is rendered
    Vec3 target_position = m_kart->getTrans()(Vec3(0, 0, 1));
    m_camera->setTarget(target_position.toIrrVector());
    m_camera->setRotation(core::vector3df(0, 0, 0));
    m_camera->setFOV(m_fov);
}   // setInitialTransform

//-----------------------------------------------------------------------------
/** Called once per time frame to move the camera to the right position.
 *  \param dt Time step.
 */
void Camera::update(float dt)
{
    if (!m_kart)
    {
        if (race_manager->getNumLocalPlayers() < 2)
        {
            Vec3 pos(m_camera->getPosition());
        }

        return; // cameras not attached to kart must be positioned manually
    }

    if (race_manager->getNumLocalPlayers() < 2)
    {
        Vec3 heading(sin(m_kart->getHeading()), 0.0f, cos(m_kart->getHeading()));
    }
}   // update

// ----------------------------------------------------------------------------
/** Sets viewport etc. for this camera. Called from irr_driver just before
 *  rendering the view for this kart.
 */
void Camera::activate(bool alsoActivateInIrrlicht)
{
    s_active_camera = this;
    if (alsoActivateInIrrlicht)
    {
        irr::scene::ISceneManager *sm = irr_driver->getSceneManager();
        sm->setActiveCamera(m_camera);
        irr_driver->getVideoDriver()->setViewPort(m_viewport);
    }
}   // activate
