CLASS z_humaneval_095 DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC .

  PUBLIC SECTION.
    TYPES: BEGIN OF ty_dict_entry,
             key   TYPE string,
             value TYPE string,
           END OF ty_dict_entry,
           tt_dict TYPE TABLE OF ty_dict_entry.

    CLASS-METHODS: check_dict_case
      IMPORTING
        VALUE(dict)   TYPE tt_dict
      RETURNING
        VALUE(result) TYPE abap_bool.

  PRIVATE SECTION.

ENDCLASS.

CLASS z_humaneval_095 IMPLEMENTATION.

  METHOD check_dict_case.
    DATA: lv_state TYPE string VALUE 'start',
          lv_key   TYPE string,
          lv_upper TYPE string,
          lv_lower TYPE string.

    IF lines( dict ) = 0.
      result = abap_false.
      RETURN.
    ENDIF.

    LOOP AT dict INTO DATA(ls_entry).
      lv_key = ls_entry-key.

      IF lv_key CO '0123456789'.
        lv_state = 'mixed'.
        EXIT.
      ENDIF.

      lv_upper = to_upper( lv_key ).
      lv_lower = to_lower( lv_key ).

      IF lv_state = 'start'.
        IF lv_key = lv_upper AND lv_key <> lv_lower.
          lv_state = 'upper'.
        ELSEIF lv_key = lv_lower AND lv_key <> lv_upper.
          lv_state = 'lower'.
        ELSE.
          EXIT.
        ENDIF.
      ELSEIF ( lv_state = 'upper' AND lv_key <> lv_upper ) OR
             ( lv_state = 'lower' AND lv_key <> lv_lower ).
        lv_state = 'mixed'.
        EXIT.
      ENDIF.
    ENDLOOP.

    IF lv_state = 'upper' OR lv_state = 'lower'.
      result = abap_true.
    ELSE.
      result = abap_false.
    ENDIF.

  ENDMETHOD.

ENDCLASS.