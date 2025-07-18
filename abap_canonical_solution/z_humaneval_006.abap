CLASS z_humaneval_006 DEFINITION
  PUBLIC
  FINAL
  ABSTRACT.

  PUBLIC SECTION.
    TYPES: ty_result TYPE i. " You can keep this or remove if unused
    CLASS-METHODS parse_nested_parens
      IMPORTING paren_string  TYPE string
      RETURNING VALUE(result) TYPE int4_table.
ENDCLASS.

CLASS z_humaneval_006 IMPLEMENTATION.

  METHOD parse_nested_parens.
    DATA lv_paren_string TYPE string.
    DATA lt_results TYPE int4_table.
    DATA lt_groups TYPE TABLE OF string.
    DATA lv_depth TYPE i.
    DATA lv_max_depth TYPE i.

    lv_paren_string = paren_string.
    SPLIT lv_paren_string AT ' ' INTO TABLE lt_groups.

    LOOP AT lt_groups INTO DATA(lv_group).
      CLEAR lv_depth.
      CLEAR lv_max_depth.

      DO strlen( lv_group ) TIMES.
        DATA(lv_offset) = sy-index - 1.
        DATA(lv_char) = lv_group+lv_offset(1).
        IF lv_char = '('.
          lv_depth = lv_depth + 1.
          IF lv_max_depth < lv_depth.
            lv_max_depth = lv_depth.
          ENDIF.
        ELSEIF lv_char = ')'.
          lv_depth = lv_depth - 1.
        ENDIF.
        IF lv_depth < 0.
          lv_depth = 0.
        ENDIF.
      ENDDO.

      APPEND lv_max_depth TO lt_results.
    ENDLOOP.

    result = lt_results.
  ENDMETHOD.

ENDCLASS.
