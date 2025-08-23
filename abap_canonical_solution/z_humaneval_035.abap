CLASS z_humaneval_035 DEFINITION
  PUBLIC
  FINAL
  ABSTRACT.

  PUBLIC SECTION.
    CLASS-METHODS: max_element
      IMPORTING
        VALUE(l) TYPE int4_table
      RETURNING
        VALUE(ev_max)  TYPE i.
  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_035 IMPLEMENTATION.
  METHOD max_element.
    IF lines( l ) = 0.
      RETURN.
    ENDIF.

    ev_max = l[ 1 ].
    LOOP AT l INTO DATA(lv_element).
      IF lv_element > ev_max.
        ev_max = lv_element.
      ENDIF.
    ENDLOOP.
  ENDMETHOD.
ENDCLASS.