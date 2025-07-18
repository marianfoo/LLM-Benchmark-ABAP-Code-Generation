CLASS z_humaneval_000 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_float_table,
             value TYPE f,
           END OF ty_float_table.
    TYPES: tt_float_table TYPE TABLE OF ty_float_table.

    CLASS-METHODS: has_close_elements
      IMPORTING
        numbers       TYPE tt_float_table
        threshold     TYPE f
      RETURNING
        VALUE(result) TYPE abap_bool.

  PRIVATE SECTION.
ENDCLASS.

CLASS z_humaneval_000 IMPLEMENTATION.
  METHOD has_close_elements.
    DATA: idx      TYPE i,
          idx2     TYPE i,
          elem     TYPE f,
          elem2    TYPE f,
          distance TYPE f.

    LOOP AT numbers INTO DATA(wa_elem) FROM 1.
      idx = sy-tabix.
      elem = wa_elem-value.

      LOOP AT numbers INTO DATA(wa_elem2) FROM 1.
        idx2 = sy-tabix.
        elem2 = wa_elem2-value.

        IF idx <> idx2.
          distance = abs( elem - elem2 ).
          IF distance < threshold.
            result = abap_true.
            RETURN.
          ENDIF.
        ENDIF.
      ENDLOOP.
    ENDLOOP.

    result = abap_false.
  ENDMETHOD.
ENDCLASS.