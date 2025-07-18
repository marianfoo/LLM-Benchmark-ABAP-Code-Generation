CLASS test_z_humaneval_141 DEFINITION
  FOR TESTING
  DURATION SHORT
  RISK LEVEL HARMLESS.

  PRIVATE SECTION.
    METHODS: test_example_txt FOR TESTING,
      test_1example_dll FOR TESTING,
      test_s1sdf3_asd FOR TESTING,
      test_k_dll FOR TESTING,
      test_my16file3_exe FOR TESTING,
      test_his12file94_exe FOR TESTING,
      test_underscore_y_txt FOR TESTING,
      test_question_areya_exe FOR TESTING,
      test_slash_this_is_valid_dll FOR TESTING,
      test_this_is_valid_wow FOR TESTING,
      test_this_is_valid_txt FOR TESTING,
      test_this_is_valid_txtexe FOR TESTING,
      test_hash_this2_i4s_5valid_ten FOR TESTING,
      test_at_this1_is6_valid_exe FOR TESTING,
      test_this_is_12valid_6exe4_txt FOR TESTING,
      test_all_exe_txt FOR TESTING,
      test_i563_no_exe FOR TESTING,
      test_is3youfault_txt FOR TESTING,
      test_no_one_hash_knows_dll FOR TESTING,
      test_1i563_yes3_exe FOR TESTING,
      test_i563_yes3_txtt FOR TESTING,
      test_final_double_dot_txt FOR TESTING,
      test_final132 FOR TESTING,
      test_underscore_f4indsartal132 FOR TESTING,
      test_dot_txt FOR TESTING,
      test_s_dot FOR TESTING.

ENDCLASS.

CLASS test_z_humaneval_141 IMPLEMENTATION.

  METHOD test_example_txt.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'example.txt' )
      exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_1example_dll.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( '1example.dll' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_s1sdf3_asd.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 's1sdf3.asd' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_k_dll.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'K.dll' )
      exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_my16file3_exe.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'MY16FILE3.exe' )
      exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_his12file94_exe.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'His12FILE94.exe' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_underscore_y_txt.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( '_Y.txt' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_question_areya_exe.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( '?aREYA.exe' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_slash_this_is_valid_dll.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( '/this_is_valid.dll' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_this_is_valid_wow.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'this_is_valid.wow' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_this_is_valid_txt.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'this_is_valid.txt' )
      exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_this_is_valid_txtexe.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'this_is_valid.txtexe' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_hash_this2_i4s_5valid_ten.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( '#this2_i4s_5valid.ten' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_at_this1_is6_valid_exe.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( '@this1_is6_valid.exe' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_this_is_12valid_6exe4_txt.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'this_is_12valid.6exe4.txt' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_all_exe_txt.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'all.exe.txt' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_i563_no_exe.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'I563_No.exe' )
      exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_is3youfault_txt.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'Is3youfault.txt' )
      exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_no_one_hash_knows_dll.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'no_one#knows.dll' )
      exp = 'Yes' ).
  ENDMETHOD.

  METHOD test_1i563_yes3_exe.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( '1I563_Yes3.exe' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_i563_yes3_txtt.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'I563_Yes3.txtt' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_final_double_dot_txt.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'final..txt' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_final132.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 'final132' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_underscore_f4indsartal132.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( '_f4indsartal132.' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_dot_txt.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( '.txt' )
      exp = 'No' ).
  ENDMETHOD.

  METHOD test_s_dot.
    cl_abap_unit_assert=>assert_equals(
      act = z_humaneval_141=>file_name_check( 's.' )
      exp = 'No' ).
  ENDMETHOD.

ENDCLASS.