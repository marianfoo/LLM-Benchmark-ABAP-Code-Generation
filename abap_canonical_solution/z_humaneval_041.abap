CLASS z_humaneval_041 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    CLASS-METHODS: car_race_collision
      IMPORTING
        n             TYPE i
      RETURNING
        VALUE(result) TYPE i.

ENDCLASS.

CLASS z_humaneval_041 IMPLEMENTATION.

  METHOD car_race_collision.
    result = n ** 2.
  ENDMETHOD.

ENDCLASS.